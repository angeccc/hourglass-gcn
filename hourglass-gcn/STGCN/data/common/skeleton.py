import numpy as np

class Skeleton:

    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right) #len(list),结果是int型

        self._parents = np.array(parents) #np.array(list),变成ndarray  _parents:0-31节点.以0节点为中心向外发散，每个的父节点,图在打印的论文的后面。0节点父节点特设为-1
        self._joints_left = joints_left   #13个左边关节
        self._joints_right = joints_right
        self._compute_metadata()          #添加了self,_children,表示每个关节的子节点

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._compute_metadata()

        return valid_joints

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool) #len(一维ndarray)=32,   astype内置函数，把数组中元素变为bool型，0变False
        for i, parent in enumerate(self._parents):  #enumerate（ndarray）, i取最高维的索引，从0到31，parent取索引对应的元素值
            if parent != -1:
                self._has_children[parent] = True  #除了-1外，父节点都有子节点，这些位置变为True;
        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])  #没用到i,parent参数，重复32次，每次列表里加一个空list元素
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i) #i的父节点是parent， parent节点的子节点列表应添加i。