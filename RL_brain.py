"""
参数设定：
根据书上已经确定的：
学习率learning_rate                     0.5
折扣因子（奖励衰减）reward_decay           0.9
贪婪度e_greedy                          0.2

trace_decay未知，人为设定为0.9
"""


import numpy as np
import pandas as pd

class RL(object):
    def __init__(self, action_space, learning_rate=0.5, reward_decay=0.9, e_greedy=0.2):
        self.actions = action_space  # 动作空间
        self.lr = learning_rate      # 学习率
        self.gamma = reward_decay    # 奖励衰减
        self.epsilon = e_greedy      # 贪婪度

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):

        self.check_state_exist(observation)

        if np.random.rand() < self.epsilon:
            # 选取q最大的a
            state_action = self.q_table.loc[observation, :]
            # 随机扰动，防止q值相同的多个a
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, *args):
        pass


class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.5, reward_decay=0.9, e_greedy=0.2, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay                      # lambda值
        self.eligibility_trace = self.q_table.copy()    #

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):

        self.check_state_exist(s_)

        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict

        # 更新eligibility_trace表
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        # 更新q表
        self.q_table += self.lr * error * self.eligibility_trace

        # 衰减eligibility_trace的值
        self.eligibility_trace *= self.gamma*self.lambda_