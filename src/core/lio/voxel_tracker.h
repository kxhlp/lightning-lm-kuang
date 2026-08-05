/**
 * @file voxel_tracker.h
 * @brief 基于体素变化的动态物体检测模块
 * 
 * 核心思想：
 * - 将3D空间划分为固定大小的立方体网格（体素）
 * - 跟踪每个体素内点云的历史位置变化
 * - 如果一个体素内的点在连续帧中发生明显位移，则判定该体素内存在动态物体
 * 
 * 适用场景：
 * - 建图：过滤掉移动物体，生成干净的静态地图
 * - 定位：检测环境变化，提升定位可靠性
 */

#ifndef LIGHTNING_VOXEL_TRACKER_H
#define LIGHTNING_VOXEL_TRACKER_H

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cmath>

#include "common/eigen_types.h"
#include "common/pointcloud.h"

namespace lightning {

/**
 * @brief 体素键（用于 unordered_map 索引）
 */
struct VoxelKey {
    int x, y, z;
    
    VoxelKey() : x(0), y(0), z(0) {}
    VoxelKey(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
    
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

/**
 * @brief 体素键的哈希函数
 */
struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
        // 使用较大的质数乘积，确保哈希分布均匀
        return ((size_t)k.x * 73856093) ^ 
               ((size_t)k.y * 19349663) ^ 
               ((size_t)k.z * 83492791);
    }
};

/**
 * @brief 体素信息（单帧数据）
 */
struct VoxelInfo {
    Vec3f center = Vec3f::Zero();   // 该体素的中心位置
    int point_count = 0;            // 体素内包含的点数
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief 体素状态（历史跟踪数据）
 */
struct VoxelState {
    Vec3f center_position = Vec3f::Zero();  // 上一次观测的中心位置
    double first_seen_time = 0.0;           // 首次观测时间
    double last_seen_time = 0.0;            // 最后一次观测时间
    int visit_count = 0;                    // 被观测到的次数
    int dynamic_count = 0;                  // 被判定为动态的次数
    int static_count = 0;                   // 被判定为静态的次数
    bool is_static = false;                 // 是否已确认为静态
    double lifetime = 0.0;                  // 存在时间（秒）
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief 动态物体检测配置选项
 */
struct VoxelTrackerOptions {
    float resolution = 0.3f;               // 体素分辨率 (m)
    float speed_threshold = 0.3f;          // 移动速度阈值 (m/s)
    int min_hits_for_static = 3;           // 判定为静态的最小命中次数
    int max_stale_frames = 100;            // 最大无观测帧数
    double max_stale_time = 10.0;          // 最大无观测时间 (s)
    double transient_threshold = 2.0;      // 短暂存在阈值 (s)
    bool enable_temporal_filtering = true;  // 是否启用时序滤波
    int temporal_window_size = 3;          // 时序滤波窗口大小
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief 动态点检测结果
 */
struct DynamicDetectionResult {
    std::vector<char> is_dynamic;           // 每个点是否为动态点
    int dynamic_point_count = 0;            // 动态点数量
    int static_point_count = 0;             // 静态点数量
    double dynamic_ratio = 0.0;             // 动态点比例
    std::vector<VoxelKey> dynamic_voxels;   // 动态体素列表（用于调试）
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief 基于体素变化的动态物体检测器
 * 
 * 使用方法：
 * @code
 *   VoxelTracker::Options options;
 *   options.resolution = 0.3f;      // 体素大小
 *   options.speed_threshold = 0.3f; // 移动速度阈值
 *   
 *   VoxelTracker tracker(options);
 *   
 *   // 每帧调用
 *   auto result = tracker.DetectAndFilter(cloud, pose, timestamp);
 *   
 *   // 使用过滤后的点云
 *   CloudPtr static_cloud(new PointCloudType());
 *   for (size_t i = 0; i < cloud->size(); ++i) {
 *       if (!result.is_dynamic[i]) {
 *           static_cloud->push_back(cloud->points[i]);
 *       }
 *   }
 * @endcode
 */
class VoxelTracker {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    /**
     * @brief 构造函数
     * @param options 配置参数
     */
    explicit VoxelTracker(const VoxelTrackerOptions& options);
    
    /**
     * @brief 默认构造函数
     */
    VoxelTracker() : VoxelTracker(VoxelTrackerOptions{}) {}
    
    ~VoxelTracker() = default;
    
    /**
     * @brief 主检测函数：检测并标记动态点
     * @param cloud 当前帧点云
     * @param pose 当前帧位姿（SE3）
     * @param timestamp 当前时间戳（秒）
     * @return 动态点检测结果
     */
    DynamicDetectionResult DetectAndFilter(const CloudPtr& cloud, 
                                          const SE3& pose, 
                                          double timestamp);
    
    /**
     * @brief 简化的检测函数，直接返回动态点掩码
     * @param cloud 当前帧点云
     * @param pose 当前帧位姿
     * @param timestamp 时间戳
     * @return 动态点掩码（true=动态点）
     */
    std::vector<char> DetectDynamicPoints(const CloudPtr& cloud,
                                          const SE3& pose,
                                          double timestamp);
    
    /**
     * @brief 过滤点云，移除动态点
     * @param cloud 输入点云
     * @param pose 位姿
     * @param timestamp 时间戳
     * @return 过滤后的静态点云
     */
    CloudPtr FilterStaticPoints(const CloudPtr& cloud,
                               const SE3& pose,
                               double timestamp);
    
    /**
     * @brief 重置跟踪器状态
     */
    void Reset();
    
    /**
     * @brief 获取当前跟踪的体素数量
     */
    size_t GetTrackedVoxelCount() const { return voxel_states_.size(); }
    
    /**
     * @brief 获取配置参数
     */
    const VoxelTrackerOptions& GetOptions() const { return options_; }
    
    /**
     * @brief 设置配置参数
     */
    void SetOptions(const VoxelTrackerOptions& options) { options_ = options; }
    
    /**
     * @brief 获取统计信息
     */
    struct Statistics {
        int total_voxels;
        int static_voxels;
        int dynamic_voxels;
        int new_voxels;
        int disappeared_voxels;
    };
    Statistics GetStatistics() const;

private:
    /**
     * @brief 将点云体素化
     */
    std::unordered_map<VoxelKey, VoxelInfo, VoxelKeyHash> 
    Voxelize(const CloudPtr& cloud, const SE3& pose);
    
    /**
     * @brief 计算体素键
     */
    VoxelKey ComputeVoxelKey(const Vec3f& point) const;
    
    /**
     * @brief 计算点的世界坐标
     */
    Vec3f TransformToWorld(const PointType& pt, const SE3& pose) const;
    
    /**
     * @brief 清理过旧的体素记录
     */
    void CleanupStaleVoxels(double current_time);
    
    /**
     * @brief 时序滤波：基于历史帧平滑动态检测结果
     */
    std::vector<char> TemporalFiltering(const std::vector<char>& current_detection,
                                        const CloudPtr& cloud,
                                        const SE3& pose);

private:
    VoxelTrackerOptions options_;
    
    /// 体素状态历史
    std::unordered_map<VoxelKey, VoxelState, VoxelKeyHash> voxel_states_;
    
    /// 时序滤波历史
    std::vector<std::pair<double, std::vector<char>>> temporal_history_;
    
    /// 上一次时间戳
    double last_timestamp_ = 0.0;
    
    /// 帧计数
    int frame_count_ = 0;
};

}  // namespace lightning

#endif  // LIGHTNING_VOXEL_TRACKER_H
