# -*- coding: utf-8 -*-
import rice
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import time
import csv

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


class CS:
    def __init__(self, size, x_max, dim):
        self.size = size  # 种群个数
        self.x_max = x_max  # 边界值
        self.dim = dim  # 维度

        # 一些影响的参数
        self.beta = 1.5  # 来维飞行中beta
        self.lamuda = 0.05  # 步长系数
        self.pa = 0.75  # 被发现的概率
        self.sigma_v = 1  # 来维飞行中 sigma_v
        self.sigma_u = (math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2) / (
                math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2)))) ** (1 / self.beta)

        # 种群初始化
        self.nest = [Particle(self.x_max, self.x_max, self.dim) for _ in range(self.size)]
        # 默认第一个为最优解
        self.best_position = self.nest[0].get_pos()  # 种群最优位置
        self.best_fitness_val = fit_fun(self.best_position)  # 最佳位置适应值
        self.get_best_nest(self.nest)

    def set_best_fitness_val(self, val):
        self.best_fitness_val = val

    def get_best_fitness_val(self):
        return self.best_fitness_val

    def set_best_position(self, new_position):
        self.best_position = new_position

    def get_best_position(self):
        return self.best_position

    def simple_bounds(self, nest):
        """
        将所有的种群的位置进行限制, 限制在内部
        """
        for part in nest:
            pos = part.get_pos()[0]
            pos[pos > self.x_max] = self.x_max
            pos[pos < -self.x_max] = -self.x_max

    def get_new_nest_with_levy(self):
        """
        使用来维飞行进行新一轮的优化, 返回优化后的内容
        """
        new_nest = self.nest.copy()
        cur_nest = self.nest.copy()
        for i in range(len(cur_nest)):
            s = cur_nest[i].get_pos()
            u = np.random.normal(0, self.sigma_u, 1)
            v = np.random.normal(0, self.sigma_v, 1)
            Ls = u / ((abs(v)) ** (1 / self.beta))  # 步长
            # print("ls:%s"%(Ls))
            stepsize = self.lamuda * Ls * (s[0] - self.best_fitness_val)  # 优化后的步长
            # print("stepsize: %s" % stepsize)
            s[0] = s[0] + stepsize * np.random.randn(1, len(s[0]))  # 产生满足正态分布的序列
            new_nest[i].set_pos(s)
        self.simple_bounds(new_nest)
        return new_nest

    def empty_nests(self):
        """
        按照Pa值进行抛弃巢穴, 然后将巢穴随机打乱
        :return:
        """
        new_nest = self.nest.copy()
        nest1 = self.nest.copy()
        nest2 = self.nest.copy()
        rand_m = self.pa - np.random.rand(self.size, self.nest[0].get_pos().shape[0], self.nest[0].get_pos().shape[1])
        rand_m = np.heaviside(rand_m, 0)
        np.random.shuffle(nest1)
        np.random.shuffle(nest2)
        for i in range(self.size):
            stepsize = np.random.rand(1) * (nest1[i].get_pos() - nest2[i].get_pos())
            new_pos = self.nest[i].get_pos() * rand_m[i][0] * stepsize
            new_nest[i].set_pos(new_pos)
        self.simple_bounds(new_nest)
        return new_nest

    def PSO_update_P_a_V(self, PSO_best_position, PSO_best_fitness_value):
        """
        使用PSO的位置和适应值更新自身的最佳位置和适应值
        :param PSO_best_position:
        :param PSO_best_fitness_value:
        :return:
        """
        if PSO_best_fitness_value < self.get_best_fitness_val():
            self.set_best_position(PSO_best_position)
            self.set_best_fitness_val(PSO_best_fitness_value)

    def get_best_nest(self, new_nest):
        """
        1. 判断是否更新当前位置
        2. 更新 最佳位置 与 最佳适应值
        :param new_nest:
        :return:
        """
        for i in range(self.size):
            temp1 = fit_fun(self.nest[i].get_pos())
            temp2 = fit_fun(new_nest[i].get_pos())
            if temp1 >= temp2:
                self.nest[i].set_pos(new_nest[i].get_pos())
                if temp2 <= self.best_fitness_val:
                    self.set_best_fitness_val(temp2)
                    self.set_best_position(new_nest[i].get_pos())

    def pso_item_run(self):
        """
        :param PSO_best_position:
        :param PSO_best_fitness_value:
        :return:
        """
        new_nest = self.get_new_nest_with_levy()
        self.get_best_nest(new_nest)
        new_nest = self.empty_nests()
        self.get_best_nest(new_nest)

    def every_item_run(self, item):
        """
        只是进行布谷鸟算法
        :param item: 输入进来的迭代次数
        :return:
        """
        for _ in range(item):
            new_nest = self.get_new_nest_with_levy()
            self.get_best_nest(new_nest)
            new_nest = self.empty_nests()
            self.get_best_nest(new_nest)


class PSO_v04:
    def __init__(self, dim, size, iter_num, x_max, max_vel, tol, w_b_limit, w_b, best_fitness_value=float('Inf'),
                 change_rate=0.5, C1=2,
                 C2=2, W=1, cs_use_num=10):
        # PSO需要的参数
        self.C1 = C1
        self.C2 = C2
        self.W = W

        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.tol = tol  # 截至条件
        self.best_fitness_value = rice.fit_fun(w_b.reshape(1, dim))
        self.best_position = w_b.reshape(1, dim)  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.cur_item_num = -1

        # 初始化的限制
        self.w_b_limit = w_b_limit  # 限制值
        self.w_b = w_b  # 神经网络最后一次的w_b

        # 遗传算法需要的参数
        self.Change_rate = change_rate

        # 布谷鸟算法需要的参数
        self.cs_num = int(self.dim / 5) if int(self.dim / 5) > 5 else 3  # cs中群数 总数量的5%, 最少5个
        # self.cs_num = 5  # 布谷鸟数量
        self.cs_use_num = cs_use_num
        self.limit_num = 0  # 用来记录当前PSO陷入局部收敛的次数

        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim) for _ in range(self.size)]
        self.cs = CS(self.size, self.x_max, self.dim)  # 布谷鸟使用的次数

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

    def change_W(self, i, T):
        '''
        w_max = 1
        w_min = 0.3
        :param i:
        :param T:对应的周期
        :return:
        '''
        # 线形递减权值 (效果不是很好)
        # self.W = (0.3 - 1)*(self.iter_num - i)/self.iter_num + 1
        # 周期性线形下降
        self.W = ((T - i) % T) * (2 / T)

    # 遗传算法进行遗传优化
    def inherit(self, p1_id, p2_id):
        '''
        运用遗传算法的思想
        将所有的列表进行随机分类, 按照两个一类, 如果单数则一个不动
        交换律self.change_rate = 0.6
        在两个当中每一个的进行判定, 判定是否进行交换, 之后生成两个新的列表
        四个列表进行检测, 留下两个效果最好的
        :param p1_id: 一个个体的ID
        :param p2_id: 另一个个体的ID
        :return:
        '''
        kid1 = self.Particle_list[p1_id].get_pos().copy()
        kid2 = self.Particle_list[p2_id].get_pos().copy()
        # 随机判定是否进行交换, 然后交换的就直接交换了
        for i in range(self.dim):
            if random.random() <= self.Change_rate:
                kid1[0][i], kid2[0][i] = kid2[0][i], kid1[0][i]
        # p1判断是否更近了
        val1 = fit_fun(kid1)
        if val1 < self.Particle_list[p1_id].get_fitness_value():
            self.Particle_list[p1_id].set_pos(kid1)  # 效果好, 用新生成的种群进行替换
            self.Particle_list[p1_id].set_fitness_value(val1)
            self.Particle_list[p1_id].set_best_pos(kid1)
            if val1 < self.get_bestFitnessValue():
                self.set_bestFitnessValue(val1)
                self.set_bestPosition(kid1)
        # p2是否更近了
        val2 = fit_fun(kid2)
        if val1 < self.Particle_list[p2_id].get_fitness_value():
            self.Particle_list[p2_id].set_pos(kid2)  # 效果好, 用新生成的种群进行替换
            self.Particle_list[p2_id].set_fitness_value(val2)
            self.Particle_list[p2_id].set_best_pos(kid2)
            if val1 < self.get_bestFitnessValue():
                self.set_bestFitnessValue(val2)
                self.set_bestPosition(kid2)

    def inherit_range(self):
        # 打乱顺序, 输入对应的值
        l = list(range(self.size))
        random.shuffle(l)
        for i in range(int(self.size / 2)):
            self.inherit(l[i], l[i + int(self.size / 2)])

    # 使用CS算法进行优化
    def CS_update_P_a_V(self):
        if self.cs.get_best_fitness_val() < self.get_bestFitnessValue():
            self.set_bestPosition(self.cs.get_best_position())
            self.set_bestFitnessValue(self.cs.get_best_fitness_val())

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

    def CS_jumpout_limit(self, old_best_fitness, new_best_fitness):
        """
        如果PSO陷入几次局部收敛, 就使用CS算法进行跳出
        :return:
        """
        if abs(old_best_fitness - new_best_fitness) <= self.tol:
            self.limit_num += 1
        if self.limit_num == self.cs_use_num:
            self.cs.PSO_update_P_a_V(self.get_bestPosition(), self.get_bestFitnessValue())
            self.cs.pso_item_run()
            self.CS_update_P_a_V()
            self.limit_num = 0

    def update_ndim(self):
        self.init_all_part()
        old_fitness = self.get_bestFitnessValue()
        for i in range(self.iter_num):
            self.change_W(i, 10)
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.inherit_range()  # 使用遗传算法进行优化

            self.CS_jumpout_limit(old_fitness, self.get_bestFitnessValue())
            old_fitness = self.get_bestFitnessValue()
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
            print('第{}次最佳适应值为{}'.format(i, self.get_bestFitnessValue()))
            train_accuracy, train_loss, test_accuracy, test_loss = rice.train(self.get_bestPosition())
            print("Iter " + str(i + 1) + ", Testing Accuracy: " + str(test_accuracy) + ", Training Accuracy: " + str(
                train_accuracy))
            print(
                "Iter " + str(i + 1) + ", Testing Loss: " + str(test_loss) + ", Training Loss: " + str(train_loss))
            print('\n')

        return self.fitness_val_list, self.get_bestPosition()


def fun1():
    t = time.time()
    # 开始使用神经网络进行初始化
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
    pso = PSO_v04(dim=6194, size=8, iter_num=100, x_max=2, max_vel=1, tol=0.01, w_b_limit=all_w_b,
                  w_b=new_w_b_list[-1],
                  C1=1, C2=1, W=1)
    fit_var_list, best_pos = pso.update_ndim()
    print("最优位置:" + str(best_pos))
    print("最优解:" + str(fit_var_list[-1]))
    # 再使用神经网络优化
    finish, test_a_l = rice.use_part_train(best_pos, 100)
    t = time.time() - t
    test_a_l.append(t)

    # 数据记录
    with open("./data/大米分类CS-Ga-W-PSO版本.csv", 'a', newline='') as csvfile:
        cr = csv.writer(csvfile)
        cr.writerow(test_a_l)
    print("数据记录完毕")


def fun2():
    t = time.time()
    # 开始使用神经网络进行初始化
    a = create_part_x()
    all_w_b_list, _ = rice.use_part_train(a, 500)
    new_w_b_list = []
    for all_w_b in all_w_b_list:
        new_w_b_list.append(rice.w_b_to_part_x(all_w_b))
    mean_w_b_change = np.zeros([6194])
    for i in range(len(new_w_b_list) - 1):
        mean_w_b_change += new_w_b_list[i + 1] - new_w_b_list[i]
    all_w_b = new_w_b_list[-1] + mean_w_b_change / len(new_w_b_list)
    # 使用PSO进行优化
    pso = PSO_v04(dim=6194, size=8, iter_num=100, x_max=2, max_vel=1, tol=0.01, w_b_limit=all_w_b,
                  w_b=new_w_b_list[-1],
                  C1=1, C2=1, W=1)
    fit_var_list, best_pos = pso.update_ndim()
    print("最优位置:" + str(best_pos))
    print("最优解:" + str(fit_var_list[-1]))
    # 再使用神经网络优化
    finish, test_a_l = rice.use_part_train(best_pos, 400)
    t = time.time() - t
    test_a_l.append(t)

    # 数据记录
    with open("./data/大米分类CS-Ga-W-PSO_前500后400版本.csv", 'a', newline='') as csvfile:
        cr = csv.writer(csvfile)
        cr.writerow(test_a_l)
    print("数据记录完毕")


if __name__ == '__main__':
    fun1()  # 前100后100
    # fun2() # 前500后400
