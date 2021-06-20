# -*- coding: utf-8 -*-
import rice
import matplotlib.pyplot as plt
import random
import numpy as np

create_part_x = rice.create_part_x
fit_fun = rice.fit_fun


class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim):
        self.__pos = np.zeros((1, dim))  # 粒子的位置, 使用权重初始化函数进行替换
        self.__vel = np.random.uniform(-max_vel, max_vel, (1, dim))  # 粒子的速度
        self.__bestPos = np.zeros((1, dim))  # 粒子最好的位置
        self.__fitnessValue = fit_fun(self.__pos)  # 适应度函数值

    def set_pos(self, value):
        self.__pos = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, value):
        self.__bestPos = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, value):
        self.__vel = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


# pso = PSO(4, 5, 10000, 30, 60, 1e-4, C1=2, C2=2, W=1)
class PSO_v01:
    def __init__(self, dim, size, iter_num, x_max, max_vel, tol, w_b_limit, w_b, best_fitness_value=float('Inf'), C1=2,
                 C2=2, W=1):
        # PSO需要的参数
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.Change_rate = 0.2

        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.tol = tol  # 截至条件
        self.w_b = w_b  # 最后一次的w_b
        self.w_b_limit = w_b_limit  # 梯度平均值, 用来进行优化

        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim) for _ in range(self.size)]
        # 中间需求
        self.best_fitness_value = rice.fit_fun(w_b.reshape(1, dim))
        self.best_position = w_b.reshape(1, dim)  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, value):
        self.best_position = value

    def get_bestPosition(self):
        return self.best_position

    def init_all_part(self):
        """通过收到的权值进行种群的优化"""
        # 训练好的w_b作为最后一个个体的起始位置
        self.Particle_list[-1].set_pos(self.w_b.reshape([1, self.dim]))
        # 其余种群按照正态分布进行
        for i in range(self.dim):
            w_normal = rice.create_np_nor(self.size - 1, self.w_b_limit[i], 0.3)
            for p_id in range(self.size - 1):
                self.Particle_list[p_id].get_pos()[0][i] = w_normal[p_id]

    # 更新速度
    def update_vel(self, part):
        vel_value = self.W * part.get_vel() + self.C1 * np.random.rand() * (part.get_best_pos() - part.get_pos()) \
                    + self.C2 * np.random.rand() * (self.get_bestPosition() - part.get_pos())
        vel_value[vel_value > self.max_vel] = self.max_vel
        vel_value[vel_value < -self.max_vel] = -self.max_vel
        part.set_vel(vel_value)

    # 更新位置
    def update_pos(self, part):
        pos_value = part.get_pos() + part.get_vel()
        part.set_pos(pos_value)
        value = fit_fun(part.get_pos())
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            part.set_best_pos(pos_value)
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            self.set_bestPosition(pos_value)

    def update_ndim(self):
        self.init_all_part()
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
            print('第{}次最佳适应值为{}'.format(i, self.get_bestFitnessValue()))
            train_accuracy, train_loss, test_accuracy, test_loss = rice.train(self.get_bestPosition())
            print("Iter " + str(i + 1) + ", Testing Accuracy: " + str(test_accuracy) + ", Training Accuracy: " + str(
                train_accuracy))
            print(
                "Iter " + str(i + 1) + ", Testing Loss: " + str(test_loss) + ", Training Loss: " + str(train_loss))
            print('\n')

        return self.fitness_val_list, self.get_bestPosition()


if __name__ == '__main__':
    # np.random.seed(10)
    # random.seed(10)
    # rice.tf.set_random_seed(1)
    import time

    t = time.time()  # 用来记录时间
    # 开始是神经网络给初始值
    a = create_part_x()
    all_w_b_list, _ = rice.use_part_train(a, 100)
    new_w_b_list = []
    for all_w_b in all_w_b_list:
        new_w_b_list.append(rice.w_b_to_part_x(all_w_b))
    mean_w_b_change = np.zeros([6194])
    for i in range(len(new_w_b_list) - 1):
        mean_w_b_change += new_w_b_list[i + 1] - new_w_b_list[i]
    all_w_b = new_w_b_list[-1] + mean_w_b_change / len(new_w_b_list)
    # 使用PSO进行优化
    pso = PSO_v01(dim=6194, size=8, iter_num=100, x_max=2, max_vel=1, tol=0.01, w_b_limit=all_w_b, w_b=new_w_b_list[-1],
                  C1=1, C2=1, W=1)
    fit_var_list, best_pos = pso.update_ndim()
    print("最优位置:" + str(best_pos))
    print("最优解:" + str(fit_var_list[-1]))
    # 再使用神经网络进行优化
    finish_list, test_a_l = rice.use_part_train(best_pos, 100)
    t = time.time() - t
    test_a_l.append(t)
    # 记录数据
    import csv

    with open("大米分类PSO-CNN.csv", 'a', newline='') as csvfile:
        cr = csv.writer(csvfile)
        cr.writerow(test_a_l)
    print("数据记录完毕")
