import numpy as np
import pprint
from scipy import interpolate

class GraphMatrix(object):
    def __init__(self, nodes, edges, p, N =1, S=1, I=50,  X=5, is_dir=False):
        self.nodes = nodes # ndarray
        self.edges = edges
        self.graph = np.zeros((nodes.shape[0], nodes.shape[0]), dtype=float)
        self.p = p # 关联点个数
        self.corr_edges = self.correlated_edges()
        # 边捆绑参数
        self.N = N
        self.S = S
        self.I = I
        self.X = X

        for (x, y) in edges:
            self.graph[x - 1, y - 1] = 1
            if not is_dir:
               self.graph[y - 1, x - 1] = 1

    def __str__(self):
        return str('n'.join([str(i) for i in self.graph]))

    def calc_dis(self, n1, n2):  # 输入具体坐标 例：n1:[12.11, 18,11]
        return np.sqrt(np.sum((n1 - n2) ** 2))

    def correlated_nodes(self, node, index): # 例：node:[12.11, 18,11]; index:点的下标
        dis = []
        for i in range(self.nodes.shape[0]):
            if i != index:
                dis.append((self.calc_dis(node, self.nodes[i, :]), i))
        dis.sort()
        return dis[: self.p]  # list (dis, node_index)

    def correlated_edges(self):
        edges_set = set(self.edges)
        edges = self.edges
        edges_iter = iter(edges)
        edge = next(edges_iter)
        corr_edges = []
        while(edge):
            if edge in edges_set:
                edge = np.array(edge)
                node_begin_corr = self.correlated_nodes(self.nodes[edge[0], :], edge[0])
                node_end_corr = self.correlated_nodes(self.nodes[edge[1], :], edge[1])
                corr_edge_by_node = []
                for i in range(self.p):
                    for j in range(self.p):
                        if node_begin_corr[i][1] != node_end_corr[j][1]:
                            new_edge_1 = (node_begin_corr[i][1], node_end_corr[j][1])
                            new_edge_2 = (node_end_corr[j][1], node_begin_corr[i][1])
                            if new_edge_1 in edges_set:
                                corr_edge_by_node.append(new_edge_1)
                                # print(len(edges_set))

                                # print(i, j)
                                # print('node_begin_corr:', node_begin_corr)
                                # print('node_end_corr:', node_end_corr)
                                edges_set.remove(new_edge_1)
                            elif new_edge_2 in edges_set:
                                corr_edge_by_node.append(new_edge_2)
                                # print(len(edges_set))

                                # print(i, j)
                                # print('node_begin_corr:', node_begin_corr)
                                # print('node_end_corr:', node_end_corr)
                                edges_set.remove(new_edge_2)
                if corr_edge_by_node:
                    corr_edges.append(corr_edge_by_node)
            try:
                edge = next(edges_iter)
            except:
                print('length of corr_edges:', len(corr_edges))
                break
        return corr_edges

    def corre_node_coef(self, x_i, x_q, q_index):
        sigma_x_q = self.correlated_nodes(x_q, q_index)[self.p-1][0]
        return np.exp(-0.5*np.sum((x_i - x_q) ** 2)/(sigma_x_q**2))/np.sqrt(2*np.pi)

    def corre_edge_coef(self, t_f, t_i):
        for i in range(len(self.corr_edges)):
            if t_f in self.corr_edges[i] or (t_f[1], t_f[0]) in self.corr_edges[i]:
                assert t_i in self.corr_edges[i] or (t_i[1], t_i[0]), 'Edge t_f and Edge t_i not in a same corre_edge set!'
                (x_df_idx, x_of_idx) = t_f
                (x_di_idx, x_oi_idx) = t_i
                if x_di_idx in np.array(self.correlated_nodes(self.nodes[x_df_idx, :], x_df_idx))[:, 1]:
                    return self.corre_node_coef(self.nodes[x_di_idx, :], self.nodes[x_df_idx, :], x_df_idx) * \
                           self.corre_node_coef(self.nodes[x_oi_idx, :], self.nodes[x_of_idx, :], x_of_idx)
                elif x_di_idx in np.array(self.correlated_nodes(self.nodes[x_of_idx, :], x_df_idx))[:, 1]:
                    return self.corre_node_coef(self.nodes[x_oi_idx, :], self.nodes[x_df_idx, :], x_df_idx) * \
                           self.corre_node_coef(self.nodes[x_di_idx, :], self.nodes[x_of_idx, :], x_of_idx)
                else:
                    print('not in a set!')

    def control_nodes_init(self, edges):  # [[x1, y1], [x2, y2], [x3, y3] ... ]
        N = self.N
        eps = 1e-3
        new_edge_array = np.empty((edges.shape[0], N+2, 2))
        print(new_edge_array.shape)
        for i in range(edges.shape[0]):
            x = np.array(edges[i])[:, 0]
            x = np.concatenate((x, [x[-1]+eps, x[-1]+2*eps]))# 保证最后一个坐标算进去
            y = np.array(edges[i])[:, 1]
            y = np.concatenate((y, [y[-1]+eps, y[-1]+2*eps]))
            f = interpolate.interp1d(x, y, kind='cubic')
            x_min, x_max = np.min(x), np.max(x)
            x_new = np.arange(x_min, x_max, (x_max-x_min)/(N+1))[1:]
            y_new = f(x_new)
            x_new = x_new.reshape((-1,1))
            y_new = y_new.reshape((-1,1))
            new_edge = np.hstack((x_new, y_new))  # 水平拼接
            #左端点加进去
            new_edge_array[i, 0, 0] = np.array(edges[i])[:, 0][0]
            new_edge_array[i, 0, 1] = np.array(edges[i])[:, 1][0]
            #控制点加进去
            new_edge_array[i, 1:-1, :] = new_edge
            #右端点加进去
            new_edge_array[i, -1, 0] = np.array(edges[i])[:, 0][-1]
            new_edge_array[i, -1, 1] = np.array(edges[i])[:, 1][-1]
        return new_edge_array # shape: edge_id, nodes(N+2), 2

    def edge_binding(self):
        # N, S, I, X = self.N, self.S, self.I, self.X
        edge_array = np.empty((edges.shape[0], 2, 2))
        corr_edges = self.corr_edges
        for n in range(self.X):
            for i in range(self.I):
                for edges_set in corr_edges:
                    for edge in edges_set:
                        edge_array[i, 0, :] = np.array(self.nodes[edge[0]])
                        edge_array[i, -1, :] = np.array(self.nodes[edge[1]])
                    self.control_nodes_init(edge_array)
                    assert self.N == edge_array.shape[1]-2 , 'control nodes num is wrong!'
                    for edge_idx in range(edge_array.shape[0]):
                        for ctrl_idx in range(1, self.N+1):
                            edge_array[edge_idx, ctrl_idx, ]






layerout_tsne = np.load('layerout_tsne.npy')
# print(layerout_tsne.shape)
# 取前1000个点
layerout_tsne = layerout_tsne[:1000, :, :]
layerout_tsne = np.reshape(layerout_tsne, (-1, 2))
# print(layerout_tsne.shape)
index = np.zeros((layerout_tsne.shape[0],layerout_tsne.shape[0]))
# print(index.shape)
nodes = layerout_tsne
stride = 4
edges = []
for i in range(0, layerout_tsne.shape[0], stride):
    for j in range(stride-1):
        edges.append((i+j, i+j+1))
# print(len(edges))
p = 100 # 关联点个数
g = GraphMatrix(nodes, edges, p, N =4, S=1, I=50,  X=5, is_dir=True)
# a = g.correlated_nodes(nodes[0], 0)
# corr_edges = g.correlated_edges()
# pprint.pprint(corr_edges)
# print('length of corr_edges:', len(corr_edges))
# np.save('corr_edges_{}.npy'.format(p), corr_edges)
# coef = g.corre_edge_coef((718,719), (14,15))
# print(coef)
# g.control_nodes_init(np.array([[[1.0, 1.0], [10.0, 10.0], [15.0, 15.0]], [[10.0, 10.0], [20.0, 20.0], [15.0, 15.0]]]))
g.edge_binding()


