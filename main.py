#  Regression Tree - 1 input param
import typing as tp
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

def read_data(path: str) -> tp.List[tp.Tuple[int, int]]:
    with open(path, 'r') as f:
        content = f.read()
    dataset = [(int((i.split())[0]), int((i.split())[1])) for i in content.splitlines()]
    return dataset

def SSR_for_mean(dataset: tp.List[tp.Tuple[int, int]], dot: int) -> tp.Tuple[int, int]:
    data_y = [dataset[i][1] for i in range(dot+1, len(dataset))]
    mean = np.mean(np.array(data_y))
    ssr = sum([(dataset[i][1]-mean)**2 for i in range(len(dataset))])

    return ssr
def find_node(dataset: tp.List[tp.Tuple[int, int]]) -> int:
    s_min = SSR_for_mean(dataset, 0)
    node = dataset[0][0]
    for i in range(1, len(dataset)-1):
        s = SSR_for_mean(dataset, i)
        if s < s_min:
            s_min = s
            node = dataset[i][0]
    return node
GLOBAL = []
def build_tree(dataset: tp.List[tp.Tuple[int, int]], tree=[], node=0, dots_in_group: int=3):
    left = [i for i in dataset if i[0] <= node]
    right = [i for i in dataset if i[0] > node]
    if len(right) <= dots_in_group and len(left) <= dots_in_group:
        return [node, left, right]
    elif len(right) <= dots_in_group:
        return [node,  build_tree(left, node=find_node(left)), right]
    elif len(left) <= dots_in_group:
        return [node, left, build_tree(right, node=find_node(right))]
    else:
        return [node, build_tree(left, node=find_node(left)), build_tree(right, node=find_node(right))]

def split_train_test(dataset: tp.List[tp.Tuple[int, int]]):
    pass

def classify(tree, dot: tp.Tuple[int, int]):
    left = tree[1]
    right = tree[2]
    node = tree[0]
    if dot[0] <= node:
        if isinstance(left[0], tuple):
            y = [i[1] for i in left]
            return st.mean(y)
        else:
            return classify(left, dot)
    elif dot[0] > node:
        if isinstance(right[0], tuple):
            y = [i[1] for i in right]
            return st.mean(y)
        else:
            return classify(right, dot)








if __name__ == "__main__":
    dataset = read_data('dataset.txt')
    test = dataset[1:]
    X = [i[0] for i in dataset]
    y = [i[1] for i in dataset]
    root = find_node(test)
    print(root)
    tr = build_tree(test, node=find_node(test), dots_in_group=3)
    print(tr)
    print(f"Predicted value for {dataset[8]} is {classify(tr, dataset[8])}")
    #print(tr)
    #plt.plot(X, y, 'r*', linestyle='')
    #plt.show()



