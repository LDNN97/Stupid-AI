import gym
import copy
import time
import numpy as np
# import multiprocessing as mp

GameName = 'FetchReach-v1'
SIGMA = 1
POPSIZE = 30
MAXGEN = 100


class Env(object):
    def __init__(self, name, max_step, max_reward):
        self.name = name
        self.f = gym.make(name)
        self.n_in = self.f.observation_space['observation'].shape[0] + \
            self.f.observation_space['desired_goal'].shape[0]
        self.n_out = self.f.action_space.shape[0] - 1

        self.max_step = max_step
        self.max_reward = max_reward

    def show(self, nn):
        s = self.f.reset()
        s = np.append(s['observation'], s['desired_goal'])
        done = False
        while not done:
            a = nn.get_action(s)
            s, _, done, _ = self.f.step(a)
            self.f.render()
            s = np.append(s['observation'], s['desired_goal'])
            if done:
                break
            print(done)
        time.sleep(5)
        self.f.close()

    @staticmethod
    def evaluate(nn, n_id=None, seed=None):
        nnn = copy.deepcopy(nn)
        if seed is not None:
            np.random.seed(seed)
            noise = ES.mirror(n_id) * SIGMA * np.random.randn(len(nn.layer))
            nnn.modify_params(noise)
        s = env.f.reset()
        s = np.append(s['observation'], s['desired_goal'])
        reward = 0
        for step in range(env.max_step):
            a = nnn.get_action(s)
            s, r, done, _ = env.f.step(a)
            s = np.append(s['observation'], s['desired_goal'])
            reward += r
            if done:
                break
        return reward


class NeuralNetwork(object):
    def __init__(self, n_in, n_hide, n_out):
        self.shape = []
        self.shape.append([n_in, n_hide])
        self.shape.append([n_hide, n_hide])
        self.shape.append([n_hide, n_out])
        self.layer = np.random.randn(n_in * n_hide + n_hide +
                                     n_hide * n_hide + n_hide +
                                     n_hide * n_out + n_out) * SIGMA

        self.v = np.zeros_like(self.layer)
        self.lr, self.mom = 0.05, 0.9

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def reshape(self):
        params, count = [], 0
        for shape in self.shape:
            ws, we = count, count + shape[1] * shape[0]
            bs, be = we, we + shape[1]
            params.append([self.layer[ws:we].reshape((shape[1], shape[0])),
                           self.layer[bs:be].reshape((shape[1], 1))])
            count += shape[0] * shape[1] + shape[1]
        return params

    def forward(self, state):
        nn = self.reshape()
        x = state.reshape(len(state), 1)
        x = self.sigmoid(np.dot(nn[0][0], x) + nn[0][1])
        x = self.sigmoid(np.dot(nn[1][0], x) + nn[1][1])
        x = np.dot(nn[2][0], x) + nn[2][1]
        return x

    def get_action(self, x):
        y = self.forward(x)
        # y = [y1, y2, y3]
        y = y / np.sqrt(np.sum(np.power(y, 2)))
        y = np.append(y, 0)
        return y

    def modify_params(self, delta):
        self.layer += delta

    def update_params(self, grad):
        self.v = self.mom * self.v + (1 - self.mom) * grad
        self.layer += self.lr * self.v

    def save(self):
        np.save("nn.npy", [self.shape, self.layer])

    def load(self):
        [self.shape, self.layer] = np.load("nn.npy")


class ES(object):
    @staticmethod
    def mirror(n): return (-1) ** (n % 2)

    def __init__(self, popsize):
        self.popsize = popsize
        self.population = np.zeros(popsize)

        rank = np.arange(1, self.popsize + 1)
        temp = np.maximum(0, np.log(self.popsize / 2 + 1) - np.log(rank))
        self.w = temp / temp.sum() - 1 / self.popsize

    def evolution_sp(self, nn):
        self.population = np.random.randint(1, 2 ** 32 - 1, size=int(self.popsize / 2)).repeat(2)

        reward = []
        for i in range(self.popsize):
            re = Env.evaluate(nn, i, self.population[i])
            reward = np.append(reward, re)
        rank = np.argsort(reward)[::-1]

        update = np.zeros(len(nn.layer))
        for i, kid in enumerate(rank):
            np.random.seed(self.population[kid])
            update += self.mirror(kid) * self.w[i] * np.random.randn(len(nn.layer))
        gradients = update / (self.popsize * SIGMA)
        nn.update_params(gradients)

        return np.average(reward)

    # def evolution_mp(self, nn, env):
    #     workers, reward = [], []
    #     pool = mp.Pool(processes=mp.cpu_count())
    #     for i in range(self.popsize):
    #         workers.append(pool.apply_async(Env.evaluate, (env, nn, i, self.population[i])))
    #     pool.close(), pool.join()
    #     reward = [w.get() for w in workers]
    #     rank = np.argsort(reward)[::-1]
    #
    #     update = np.zeros(len(nn.layer))
    #     for i, kid in enumerate(rank):
    #         np.random.seed(self.population[kid])
    #         update += self.mirror(kid) * self.w[i] * np.random.randn(len(nn.layer))
    #     gradients = update / (self.popsize * SIGMA)
    #     nn.update_params(gradients)
    #
    #     return np.average(reward)


def learning():
    for gen in range(MAXGEN):
        ts = time.time()
        net_ar = es.evolution_sp(net)
        te = time.time()
        print('Gen: ', gen,
              ' net_ar: %.3f' % net_ar,
              ' t: %.3f' % (te - ts))
        if net_ar > env.max_reward:
            break
    net.save()


def show():
    env.show(net)


if __name__ == "__main__":
    env = Env(GameName, 10000, 0)
    net = NeuralNetwork(env.n_in, 30, env.n_out)
    es = ES(POPSIZE)

    learning()
    show()
