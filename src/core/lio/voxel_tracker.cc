#include "core/lio/voxel_tracker.h"

#include <algorithm>
#include <numeric>
#include <cassert>

#include <glog/logging.h>

namespace lightning {

VoxelTracker::VoxelTracker(const VoxelTrackerOptions& options) : options_(options) {
    LOG(INFO) << "VoxelTracker initialized with resolution=" << options_.resolution 
              << "m, speed_threshold=" << options_.speed_threshold << "m/s";
}

VoxelKey VoxelTracker::ComputeVoxelKey(const Vec3f& point) const {
    const float inv_res = 1.0f / options_.resolution;
    return VoxelKey(
        static_cast<int>(std::floor(point.x() * inv_res)),
        static_cast<int>(std::floor(point.y() * inv_res)),
        static_cast<int>(std::floor(point.z() * inv_res))
    );
}

Vec3f VoxelTracker::TransformToWorld(const PointType& pt, const SE3& pose) const {
    Vec3d point_body(pt.x, pt.y, pt.z);
    Vec3d world_pt = pose * point_body;
    return world_pt.cast<float>();
}

std::unordered_map<VoxelKey, VoxelInfo, VoxelKeyHash> 
VoxelTracker::Voxelize(const CloudPtr& cloud, const SE3& pose) {
    std::unordered_map<VoxelKey, VoxelInfo, VoxelKeyHash> voxel_map;
    voxel_map.reserve(cloud->size() / 10);  // 预分配
    
    for (const auto& pt : cloud->points) {
        // 坐标变换到世界坐标系
        Vec3f world_pt = TransformToWorld(pt, pose);
        
        // 计算体素键
        VoxelKey key = ComputeVoxelKey(world_pt);
        
        // 累加到对应体素
        auto it = voxel_map.find(key);
        if (it == voxel_map.end()) {
            VoxelInfo info;
            info.center = world_pt;
            info.point_count = 1;
            voxel_map.emplace(key, info);
        } else {
            it->second.center += world_pt;
            it->second.point_count++;
        }
    }
    
    // 计算每个体素的中心位置
    for (auto& kv : voxel_map) {
        if (kv.second.point_count > 0) {
            kv.second.center /= kv.second.point_count;
        }
    }
    
    return voxel_map;
}

std::vector<char> VoxelTracker::DetectDynamicPoints(const CloudPtr& cloud,
                                                     const SE3& pose,
                                                     double timestamp) {
    auto result = DetectAndFilter(cloud, pose, timestamp);
    return result.is_dynamic;
}

DynamicDetectionResult VoxelTracker::DetectAndFilter(const CloudPtr& cloud,
                                                     const SE3& pose,
                                                     double timestamp) {
    DynamicDetectionResult result;

    if (cloud->empty()) {
        result.is_dynamic = std::vector<char>(cloud->size(), 0);
        return result;
    }

    const size_t num_points = cloud->size();
    result.is_dynamic.resize(num_points, 0);

    // Step 1: 体素化当前帧点云
    auto current_voxels = Voxelize(cloud, pose);

    // Step 2: 检测每个体素的变化
    std::unordered_set<VoxelKey, VoxelKeyHash> dynamic_voxel_set;
    const bool in_warmup = (frame_count_ < options_.warmup_frames);

    for (const auto& kv : current_voxels) {
        const VoxelKey& key = kv.first;
        const VoxelInfo& info = kv.second;

        auto it = voxel_states_.find(key);

        if (it == voxel_states_.end()) {
            // ========== 新体素 ==========
            // 记录新体素，但不直接标记为动态
            VoxelState state;
            state.center_position = info.center;
            state.first_seen_time = timestamp;
            state.last_seen_time = timestamp;
            state.visit_count = 1;
            state.lifetime = 0.0;
            voxel_states_.emplace(key, state);

        } else {
            // ========== 已存在体素 ==========
            VoxelState& state = it->second;

            // 计算位移和速度
            Vec3f displacement = info.center - state.center_position;
            double dt = timestamp - state.last_seen_time;
            float disp_norm = displacement.norm();

            // ● 冷启动期只更新位置，不做判定（避免第二帧雪崩）
            // ● 位移低于下界也忽略（位姿抖动/odom 噪声）
            bool confident = (state.visit_count >= options_.warmup_frames) && !in_warmup;
            float speed = (dt > 0.001 && confident) ? disp_norm / static_cast<float>(dt) : 0.0f;

            if (confident && disp_norm > options_.min_displacement && speed > options_.speed_threshold) {
                // 速度超过阈值 → 标记为动态体素
                dynamic_voxel_set.insert(key);
                state.dynamic_count++;
            } else if (confident && disp_norm <= options_.min_displacement) {
                // 几乎不动 → 倾向静态
                state.static_count++;
            }

            // 投票机制：连续 N 帧判定为动态才标记为动态
            // 用最近 N 次 dynamic_count 增量（即本次 + 前 N-1 帧）来判断
            if (state.dynamic_count >= options_.dynamic_votes_required &&
                state.dynamic_count > state.static_count) {
                dynamic_voxel_set.insert(key);
            }

            // 静态判定：连续多次稳定观测 + 没有动态计数 → 标记为静态
            if (state.static_count >= options_.min_hits_for_static &&
                state.dynamic_count == 0) {
                state.is_static = true;
            }
            // 动态可以"翻案"为静态：动态计数被大量静态观测压制
            else if (state.is_static && state.static_count > state.dynamic_count * options_.dynamic_to_static_decay) {
                // 保留静态标记
            } else if (!state.is_static &&
                       state.static_count >= options_.min_hits_for_static &&
                       state.dynamic_count * options_.dynamic_to_static_decay <= state.static_count) {
                // 累积足够静态证据后，先翻案再说
                state.is_static = true;
            }

            // 更新体素状态
            state.center_position = info.center;
            state.last_seen_time = timestamp;
            state.visit_count++;
        }
    }

    // Step 3: 检测消失的体素（用于辅助判断）
    // 仅当体素连续观测过 warmup_frames 帧以上、且确实消失时, 标记其外接体素为"潜在动态"
    for (const auto& kv : voxel_states_) {
        const VoxelKey& key = kv.first;
        if (current_voxels.find(key) == current_voxels.end()) {
            const VoxelState& st = kv.second;
            double time_since_last_seen = timestamp - st.last_seen_time;

            // 是个老体素(经历过 warmup), 且判定不够稳定, 短暂消失
            if (time_since_last_seen < options_.max_stale_time &&
                st.visit_count >= options_.warmup_frames &&
                !st.is_static &&
                st.dynamic_count > 0) {
                // 标记8邻域为潜在动态
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dz = -1; dz <= 1; dz++) {
                            VoxelKey neighbor_key{key.x + dx, key.y + dy, key.z + dz};
                            if (current_voxels.find(neighbor_key) != current_voxels.end()) {
                                dynamic_voxel_set.insert(neighbor_key);
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 4: 生成动态点掩码
    const float inv_res = 1.0f / options_.resolution;

    for (size_t i = 0; i < num_points; ++i) {
        const auto& pt = cloud->points[i];
        Vec3f world_pt = TransformToWorld(pt, pose);

        // 计算点所属的体素
        VoxelKey key(
            static_cast<int>(std::floor(world_pt.x() * inv_res)),
            static_cast<int>(std::floor(world_pt.y() * inv_res)),
            static_cast<int>(std::floor(world_pt.z() * inv_res))
        );

        // 检查是否是动态体素
        if (dynamic_voxel_set.find(key) != dynamic_voxel_set.end()) {
            result.is_dynamic[i] = 1;
        }
    }

    // Step 5: 时序滤波（可选）
    if (options_.enable_temporal_filtering) {
        result.is_dynamic = TemporalFiltering(result.is_dynamic, cloud, pose);
    }

    // Step 6: 统计结果
    result.dynamic_voxels.assign(dynamic_voxel_set.begin(), dynamic_voxel_set.end());
    result.dynamic_point_count = std::accumulate(result.is_dynamic.begin(),
                                                   result.is_dynamic.end(), 0);
    result.static_point_count = num_points - result.dynamic_point_count;
    result.dynamic_ratio = (num_points > 0) ?
                           static_cast<double>(result.dynamic_point_count) / num_points : 0.0;

    // ● 异常保护: 动态点比例过高说明检测器失效, 放弃过滤
    if (result.dynamic_ratio > options_.max_ratio_filter) {
        LOG(WARNING) << "VoxelTracker: dynamic ratio " << result.dynamic_ratio
                     << " exceeded threshold, disabling filter for this frame";
        std::fill(result.is_dynamic.begin(), result.is_dynamic.end(), 0);
        result.dynamic_point_count = 0;
        result.static_point_count = static_cast<int>(num_points);
        result.dynamic_ratio = 0.0;
    }

    // Step 7: 清理过旧的体素记录
    CleanupStaleVoxels(timestamp);

    last_timestamp_ = timestamp;
    frame_count_++;

    return result;
}

std::vector<char> VoxelTracker::TemporalFiltering(const std::vector<char>& current_detection,
                                                   const CloudPtr& cloud,
                                                   const SE3& pose) {
    // 保存当前帧检测结果到历史
    temporal_history_.push_back({last_timestamp_, current_detection});
    
    // 保持窗口大小
    while (temporal_history_.size() > static_cast<size_t>(options_.temporal_window_size)) {
        temporal_history_.erase(temporal_history_.begin());
    }
    
    // 如果历史不足，使用当前检测结果
    if (temporal_history_.size() < 2) {
        return current_detection;
    }
    
    // 时序投票：只有当连续多帧都被判定为动态时才标记为动态
    std::vector<char> filtered = current_detection;
    const size_t num_points = cloud->size();
    const int min_votes = std::min(2, static_cast<int>(temporal_history_.size()));
    
    for (size_t i = 0; i < num_points; ++i) {
        int votes = 0;
        
        // 统计历史帧中该点被判定为动态的次数
        for (const auto& history : temporal_history_) {
            if (history.second[i]) {
                votes++;
            }
        }
        
        // 当前帧判定为动态，且获得足够投票
        if (current_detection[i] && votes >= min_votes) {
            filtered[i] = 1;
        } else {
            filtered[i] = 0;
        }
    }
    
    return filtered;
}

CloudPtr VoxelTracker::FilterStaticPoints(const CloudPtr& cloud,
                                           const SE3& pose,
                                           double timestamp) {
    auto result = DetectAndFilter(cloud, pose, timestamp);
    
    CloudPtr static_cloud(new PointCloudType());
    static_cloud->reserve(cloud->size());
    
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (!result.is_dynamic[i]) {
            static_cloud->push_back(cloud->points[i]);
        }
    }
    
    return static_cloud;
}

void VoxelTracker::CleanupStaleVoxels(double current_time) {
    for (auto it = voxel_states_.begin(); it != voxel_states_.end(); ) {
        double time_since_last_seen = current_time - it->second.last_seen_time;
        
        if (time_since_last_seen > options_.max_stale_time) {
            it = voxel_states_.erase(it);
        } else {
            ++it;
        }
    }
    
    // 限制最大体素数量，防止内存无限增长
    constexpr size_t MAX_VOXEL_COUNT = 100000;
    if (voxel_states_.size() > MAX_VOXEL_COUNT) {
        LOG(WARNING) << "Voxel count exceeds limit, clearing old states";
        // 保留最近访问的体素
        std::vector<std::pair<VoxelKey, VoxelState>> sorted_states;
        sorted_states.reserve(voxel_states_.size());
        
        for (auto& kv : voxel_states_) {
            sorted_states.emplace_back(kv.first, kv.second);
        }
        
        // 按最后访问时间排序
        std::sort(sorted_states.begin(), sorted_states.end(),
            [](const auto& a, const auto& b) {
                return a.second.last_seen_time > b.second.last_seen_time;
            });
        
        // 保留前一半
        voxel_states_.clear();
        for (size_t i = 0; i < sorted_states.size() / 2; ++i) {
            voxel_states_.emplace(sorted_states[i].first, sorted_states[i].second);
        }
    }
}

void VoxelTracker::Reset() {
    voxel_states_.clear();
    temporal_history_.clear();
    last_timestamp_ = 0.0;
    frame_count_ = 0;
}

VoxelTracker::Statistics VoxelTracker::GetStatistics() const {
    Statistics stats;
    stats.total_voxels = static_cast<int>(voxel_states_.size());
    stats.static_voxels = 0;
    stats.dynamic_voxels = 0;
    stats.new_voxels = 0;
    stats.disappeared_voxels = 0;
    
    for (const auto& kv : voxel_states_) {
        if (kv.second.is_static) {
            stats.static_voxels++;
        } else if (kv.second.dynamic_count > 0) {
            stats.dynamic_voxels++;
        } else if (kv.second.visit_count == 1) {
            stats.new_voxels++;
        }
    }
    
    return stats;
}

}  // namespace lightning
