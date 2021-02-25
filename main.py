from env_agent import Field
from RL_brain import SarsaLambdaTable

MAX_TRAINING_TIME = 100


def update():
    # 训练100次
    for episode in range(MAX_TRAINING_TIME):
        # 环境初始化，得到初始状态
        observation = env.reset()

        # 根据状态确定初始动作
        action = RL.choose_action(str(observation))

        # eligibility_trace初始化
        RL.eligibility_trace *= 0

        print('11111')

        while True:
            # 获取当前动作产生的结果
            observation_, reward, done = env.step(action)

            # 由当前状态根据当前策略（q表）选取新动作
            action_ = RL.choose_action(str(observation_))

            # 学习更新策略
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 下一步
            observation = observation_
            action = action_

            # 成功或失败
            if done:
                break

    # 训练完成
    print('finished')


if __name__ == "__main__":
    # 加载环境代理与学习算法
    env = Field()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    # 学习过程
    update()

    # 将策略保存在本地文件中
    RL.q_table.to_csv('/Users/yws/Desktop/policy.csv', sep=',', header=True, index=True)
