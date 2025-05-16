import numpy as np
import matplotlib.pyplot as plt
from connective_quadtree import QuadTree, QuadTreeNode, TreeNodeState

def visualize_quadtree(quadtree: QuadTree, title: str = "QuadTree Visualization"):
    """可视化四叉树的结构和节点状态"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制原始占用网格
    ax.imshow(quadtree.root.values, cmap='binary')
    
    # 为不同状态的节点使用不同的颜色
    colors = {
        TreeNodeState.FREE: 'green',
        TreeNodeState.MIX: 'yellow',
        TreeNodeState.FULL: 'red',
        TreeNodeState.UNKNOWN: 'gray'
    }
    
    def draw_node(node: QuadTreeNode):
        if node.is_leaf:
            # 绘制叶节点
            rect = plt.Rectangle(
                (node.j, node.i),  # 左下角坐标
                node.width,        # 宽度
                node.height,       # 高度
                fill=False,
                edgecolor=colors[node.state],
                linewidth=2
            )
            ax.add_patch(rect)
            
            # 在节点中心添加状态标记
            center_x = node.j + node.width/2
            center_y = node.i + node.height/2
            ax.text(center_x, center_y, f"{node.state.name}", 
                   ha='center', va='center', color=colors[node.state])
        else:
            # 递归绘制子节点
            for child in node.children:
                draw_node(child)
    
    # 从根节点开始绘制
    draw_node(quadtree.root)
    
    # 设置标题和坐标轴
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

def main():
    # 创建一个示例占用网格
    # 0 表示障碍物，1 表示空闲空间
    grid_size = 100
    occupancy_grid = np.ones((grid_size, grid_size))
    
    # 添加一些障碍物
    # 在中心添加一个方形障碍物
    center = grid_size // 2
    size = 20
    occupancy_grid[center-size:center+size, center-size:center+size] = 0
    
    # 添加一些随机障碍物
    np.random.seed(42)
    for _ in range(10):
        x = np.random.randint(0, grid_size)
        y = np.random.randint(0, grid_size)
        r = np.random.randint(5, 15)
        occupancy_grid[max(0, x-r):min(grid_size, x+r), 
                      max(0, y-r):min(grid_size, y+r)] = 0
    
    # 创建四叉树
    min_cell_size = 10
    quadtree = QuadTree(occupancy_grid, min_cell_size=min_cell_size)
    
    # 可视化四叉树
    visualize_quadtree(quadtree, "QuadTree Structure Visualization")
    
    # 打印一些基本信息
    print(f"Total number of leaf nodes: {len(quadtree.leaf_nodes)}")
    print(f"Maximum depth: {quadtree.max_depth}")
    
    # 统计不同状态的节点数量
    state_counts = {state: 0 for state in TreeNodeState}
    for node in quadtree.leaf_nodes:
        state_counts[node.state] += 1
    
    print("\nNode states distribution:")
    for state, count in state_counts.items():
        print(f"{state.name}: {count}")

if __name__ == "__main__":
    main() 