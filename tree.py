from collections import defaultdict, Counter
import statistics
from PIL import Image
from tqdm import tqdm
import os
import random
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def display(self, level=0):
        # 确保将值转换为字符串，打印节点值
        value_str = str(self.value)  # 将 value 转换为字符串
        print(f"Level {level}: " + ' ' * level * 2 + value_str)  # 添加调试信息
        for child in self.children:
            child.display(level + 1)

def build_tree_with_levels(lines):
    if not lines:
        return None

    # 根节点为第一行
    root = TreeNode(lines[0].strip())
    node_stack = [(root, 0)]  # 堆栈存储节点及其层级

    for line in lines[1:]:
        stripped_line = line.lstrip()  # 去掉前面的空格
        level = len(line) - len(stripped_line)  # 缩进决定了层级

        parts = stripped_line.split('|')
        root_part = parts[0].strip()  # 第一个部分作为当前节点值
        new_node = TreeNode(root_part)

        # 如果有多个部分，后面的作为子节点
        if len(parts) > 1:
            for part in parts[1:]:
                child_part = part.strip()
                new_node.add_child(TreeNode(child_part))

        # 弹出堆栈中大于或等于当前层级的节点
        while node_stack and node_stack[-1][1] >= level:
            node_stack.pop()

        # 检查堆栈是否为空，然后添加新节点为父节点的子节点
        if node_stack:
            node_stack[-1][0].add_child(new_node)

        # 将新节点及其层级压入堆栈
        node_stack.append((new_node, level))

    return root

def collect_deepest_leaves(node, depth=0, max_depth_info=None):
    if max_depth_info is None:
        max_depth_info = {'max_depth': -1, 'leaf_values': {}, 'parent_values': {}}

    # 如果是叶子节点，记录叶子节点的值以及其父节点的值
    if not node.children:
        if depth > max_depth_info['max_depth']:
            max_depth_info['max_depth'] = depth
            max_depth_info['leaf_values'] = {depth: [node.value]}
            max_depth_info['parent_values'] = {depth: [node.value]}  # 假设叶子节点存储了母节点的信息
        elif depth == max_depth_info['max_depth']:
            max_depth_info['leaf_values'][depth].append(node.value)
            max_depth_info['parent_values'][depth].append(node.value)
    else:
        # 递归遍历子节点
        for child in node.children:
            collect_deepest_leaves(child, depth + 1, max_depth_info)

    return max_depth_info

# 获取每棵树的最底层叶子节点值
def get_deepest_leaf_values(root):
    max_depth_info = collect_deepest_leaves(root)
    max_depth = max_depth_info['max_depth']
    return max_depth_info['leaf_values'][max_depth] if max_depth in max_depth_info['leaf_values'] else []

# 获取母节点值
def get_parent_node_values(root):
    max_depth_info = collect_deepest_leaves(root)
    max_depth = max_depth_info['max_depth']
    return max_depth_info['parent_values'][max_depth] if max_depth in max_depth_info['parent_values'] else []

# 1. 删除最底层节点数量不一致的树
def remove_trees_with_different_deepest_node_count(roots):
    # Step 1: 计算每棵树最底层节点数量
    deepest_node_counts = [len(get_deepest_leaf_values(root)) for root in roots]

    # Step 2: 统计每个最底层节点数的出现次数
    count_frequency = Counter(deepest_node_counts)

    # Step 3: 找到出现最多的节点数（多数派）
    most_common_count, _ = count_frequency.most_common(1)[0]

    # Step 4: 过滤掉节点数与多数派不匹配的树
    filtered_roots = [root for root, count in zip(roots, deepest_node_counts) if count == most_common_count]

    return filtered_roots

def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# 获取每棵树的最底层数字叶子节点值
def get_deepest_numeric_leaf_values(root):
    max_depth_info = collect_deepest_leaves(root)
    max_depth = max_depth_info['max_depth']
    # 只提取包含数字的叶子节点
    return [v for v in max_depth_info['leaf_values'][max_depth] if is_number(v)] if max_depth in max_depth_info['leaf_values'] else []

# 2. 删除最底层叶子节点数量不一致的树
def remove_trees_with_different_deepest_numeric_leaf_values(roots):
    # Step 1: 提取每棵树的最底层数字叶子节点值
    deepest_numeric_leaf_values_list = [tuple(sorted(get_deepest_numeric_leaf_values(root))) for root in roots]

    # Step 2: 统计每种最底层数字叶子节点值集合的出现次数
    count_frequency = Counter(deepest_numeric_leaf_values_list)
    
    # Step 3: 找到最常见的数字叶子节点值集合（多数派）
    most_common_leaf_values, _ = count_frequency.most_common(1)[0]

    # Step 4: 过滤掉数字叶子节点值与多数派不匹配的树
    filtered_roots = [root for root, leaf_values in zip(roots, deepest_numeric_leaf_values_list) if leaf_values == most_common_leaf_values]

    return filtered_roots

# 提取最底层节点及其直接母节点的值
def get_parent_and_child_values(root):
    """
    从树结构中提取每棵树最底层节点及其直接母节点的值。
    """
    parent_values = []  # 存储母节点的值
    child_values = []   # 存储对应的子节点值

    # 假设树的结构是 root.children，递归遍历树的每个节点
    def traverse(node):
        if not node.children:  # 如果是叶子节点
            return

        # 当前节点是母节点，它的子节点是叶子节点
        for child in node.children:
            if not child.children:  # 只处理直接母节点和最底层叶子节点
                parent_values.append(node.value)  # 当前节点作为母节点
                child_values.append(child.value)  # 子节点
            else:
                traverse(child)  # 如果不是叶子节点，继续递归遍历

    # 开始遍历树
    traverse(root)
    
    return parent_values, child_values

# 找到最佳的基础树
def find_best_tree(roots):
    """
    在6棵树中选出一棵最优树作为基础树。这里的逻辑可以是随机选择、某种评分机制等。
    """
    # 此处假设随机选择第一棵树为基础树（可以根据其他条件选择）
    return roots[0]

# 归并母节点名称相同的子节点，删除子节点与多数派不一致的节点，并修改数值型节点
def modify_best_tree_based_on_votes(roots):
    # 归并母节点相同的子节点
    parent_to_children = defaultdict(list)

    # 收集所有母节点和子节点的值
    for root in roots:
        parent_values, child_values = get_parent_and_child_values(root)
        for parent, child in zip(parent_values, child_values):
            parent_to_children[parent].append(child)

    # 打印归并后的母节点和子节点
    print(f"归并后的母节点和子节点: {parent_to_children}")

    # 选取一棵最好的基础树（作为模板）
    best_tree = find_best_tree(roots)

    # 修改基础树中的数值型节点
    def traverse_and_modify(node):
        # 如果是叶子节点，直接返回
        if not node.children:
            return

        # 对每个子节点进行修改
        for child in node.children:
            if not child.children:  # 最底层的叶子节点
                parent_value = node.value
                child_value = child.value

                # 获取该母节点下的所有子节点值
                all_children = parent_to_children[parent_value]

                # 过滤出数值型子节点并计算平均值
                numeric_children = []
                for value in all_children:
                    if isinstance(value, (int, float)):
                        numeric_children.append(value)
                    elif isinstance(value, str) and value.isdigit():
                        numeric_children.append(int(value))
                    else:
                        try:
                            numeric_children.append(float(value))
                        except ValueError:
                            pass  # 忽略非数值型子节点

                # 计算平均值并替换数值型子节点
                if numeric_children:
                    average_value = statistics.mean(numeric_children)
                    print(f"母节点: {parent_value}, 原始子节点: {child_value}, 替换为平均值: {average_value}")
                    child.value = average_value
            else:
                traverse_and_modify(child)

    # 修改最好的树
    traverse_and_modify(best_tree)

    return best_tree


def main(out):
    roots = []
    for item in out:
        prediction = item['prediction']
        lines = prediction.split('<0x0A>')
        root = build_tree_with_levels(lines)
        roots.append(root)
        if root:
            root.display()
    roots = remove_trees_with_different_deepest_node_count(roots)
    roots = remove_trees_with_different_deepest_numeric_leaf_values(roots)
    best_tree_after_modification = modify_best_tree_based_on_votes(roots)
    best_tree_after_modification.display()


