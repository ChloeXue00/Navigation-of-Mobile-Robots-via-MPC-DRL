class ObstacleAvoidanceReward(Component):
    def step(self, action):
        # 计算机器人与障碍物的距离
        min_distance = compute_min_distance(self.env.agent.position, self.env.obstacles)
        # 如果距离过近，给予较大惩罚
        return -100 if min_distance < safe_distance else 0
