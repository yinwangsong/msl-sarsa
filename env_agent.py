"""
状态空间s：
机器人相对足球横坐标              px
机器人相对足球纵坐标              py
机器人相对足球线速度分解量         vx
机器人相对足球线速度分解量         vy
足球线速度                      vb

量化区间可依据数据范围自定

动作空间a：
以机器人正前方为编号0，顺时针8个方向分别以最大加速度运动

奖励空间r：
每一步                          -0.2
成功截球                         500
机器人或者球出界（截球失败）        -10
"""

class Field:
    def __init__(self):
        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7]
        self.n_actions = len(self.action_space)

    # 从仿真模型中获取当前状态并用量化区间法离散化
    def get_status(self):
        """
        这里与仿真环境通信，读取状态并转化为list返回
        """
        status = []
        success = 0 # 0失败1成功2在路上
        return status, success

    # 重新初始化环境并返回一个初始状态
    def reset(self):
        """
        这里与仿真环境通信，reset世界模型
        """
        # 获取初始状态
        return self.get_status()[0]

    # 在环境中执行当前动作，得到返回结果
    def step(self, action):
        """
        这里与仿真环境通信，执行当前动作
        """

        # 获取下一状态
        s_ = self.get_status()[0]
        success = self.get_status()[1]

        # 根据下一状态获取奖励
        if success == 0:
            reward = -10 - 0.2
            done = True
            s_ = 'terminal'
        elif success == 1:
            reward = 500 - 0.2
            done = True
            s_ = 'terminal'
        else:
            reward = -0.2
            done = False

        return s_, reward, done


