import numpy as np
from abc import abstractmethod


class DecisionTree:
    
    def __init__(self, X: np.array, y: np.array, depth=5,prune_data_threshold=5):
        self.meta = {
            "max_depth": depth,
            "prune_threshold":prune_data_threshold
        }
        # More interesting attributes of the tree could be added here
        self.root=DTNode.fit(X, y, 0, self)
        
    def classify(self, xs: np.array):
        if len(xs.shape) == 2:
            return np.array([self.root.eval(x) for x in xs])
        else:
            return self.root.eval(xs)
          
    def score(self,Xs:np.array,ys:np.array):
        assert len(Xs) == len(ys)
        _, counts =np.unique(self.classify(Xs)==ys,return_counts=True)
        if(len(counts) == 1):
            return 0
        return counts[1]/len(ys)
        
        
    def __str__(self):
        return "Tree: depth="+str(self.meta["max_depth"]) + "\n" + str(self.root) 


class DTNode:
    
    def __init__(self,depth):
        self.depth = depth
    
    @staticmethod
    def fit(X, y, depth: int, tree):
        assert len(X) != 0 and len(y) != 0
        if DTNode.should_split(X, y, depth, tree.meta):
            return DTLeaf(y, depth)
        rule, col = DTNode.bestRule(X, y)
        change_indecies = np.ones(shape=len(X), dtype=bool)
        for i in range(len(X)):
            change_indecies[i] = rule.apply(X[i])
        newX = X[change_indecies]
        newy = y[change_indecies]
        change_indecies = np.invert(change_indecies)
        newXO = X[change_indecies]
        newyO = y[change_indecies]
        if len(newyO) == 0:
            return DTLeaf(newy, depth)
        if len(newy) == 0:
            return DTLeaf(newyO, depth)
        # The checking of the length is just for the case that the best split
        # can't split the data any more, in which case the gain is 0,
        # and further improvements can not be made
        return DTSplit(rule, newX, newy, newXO, newyO, depth, tree)
    
    @staticmethod
    def should_split(X, y, depth, meta):
        if meta["max_depth"] <= depth:
            return True
        if meta["prune_threshold"] >= len(y):
            return True
        if len(np.unique(y)) == 1:
            return True
        #astype("<U22") is used, because axis does not work for all types
        if len(np.unique(X.astype("<U22"),axis=0)) == 1:
            return True
        return False
    
    @staticmethod
    def bestRule(X, y):
        rules = [Splitrule(X, y, i) for i in range(X.shape[1])]
        gains = []
        for rule in rules:
            trues = np.array([rule.apply(x) for x in X])
            true_ys = y[trues]
            false_ys = y[np.invert(trues)]
            gains.append(Splitrule.gain(true_ys, false_ys, y))
        col = np.argmin(gains)
        return rules[col], col
    
    @abstractmethod
    def eval(self,x):
        pass
    
    @abstractmethod
    def __str__(self):
        pass


class DTSplit(DTNode):
    
    def __init__(self, rule, newX, newy, newXO, newyO, depth, tree):
        super().__init__(depth)
        self.rule = rule
        self.node1 = DTNode.fit(newX, newy, depth+1, tree)
        self.node2 = DTNode.fit(newXO, newyO, depth+1, tree)
    
    def eval(self, x):
        if self.rule.apply(x):
            return self.node1.eval(x)
        return self.node2.eval(x)
    
    def __str__(self):
        space="|\t" * self.depth
        return space + str(self.rule)+"\n"+space+"\tthen: \n"+str(self.node1)+"\n"+space+"\t else: \n"+str(self.node2)


class DTLeaf(DTNode):
    
    def __init__(self, y,depth):
        super().__init__(depth)
        possibles, counts = np.unique(y, return_counts=True)
        self.value = possibles[np.argmax(counts)]
        
    def eval(self, x):
        return self.value
    
    def __str__(self):
        return "|\t"*self.depth + str(self.value)


class Splitrule:
    
    def __init__(self, X, y, col_index):
        assert len(X) != 0 and len(y) > 0, "The lengths should not occur in splitrule"
        assert col_index >= 0, "col_index can not be negative"
        assert len(np.unique(y)) > 1
        x=X[:, col_index]
        self.index = col_index
        self.continous = Splitrule.allNumbers(x)
        if self.continous:
            options = np.sort(np.unique(x))
        else:
            options = np.unique(x)
        self.splitval = options[np.argmin([self.attempt_split(x, y, o) for o in options])]
        
    def apply(self, x):
        if self.continous:
            return x[self.index] > self.splitval
        return (x[self.index] == self.splitval) 
            
        
    def attempt_split(self, x, y, case):
        trues =x is case if not self.continous else x > case
        y_true = y[trues]
        y_false = y[np.invert(trues)]
        return Splitrule.gain(y_true,y_false,y)
        
    @staticmethod
    def allNumbers(x):
        try:
            x.astype(np.number)
            return True
        except Exception:
            return False
        
    @staticmethod
    def gain(y_true, y_false, y):
        return (len(y_true)/len(y))*Splitrule.GINI(y_true) + (len(y_false)/len(y))*Splitrule.GINI(y_false)
        
    @staticmethod
    def GINI(y):
        if len(np.unique(y)) is 1:
            return 0
        _, counts = np.unique(y, return_counts = True)
        sumc = 0
        for c in counts:
            sumc += (c/len(y))**2
        gin = 1 - sumc
        return gin
    
    def __str__(self):
        if not self.continous:
            return "if value at %x is %s" % (self.index,self.splitval)
        return "if value at %x is greater than %s" % (self.index,self.splitval)


class RandomForest:
  
    def __init__(self,X,ys,tree_amount,max_depth=5,prune_data_threshold=10):
        subXs,subys=RandomForest.createSubsets(X,ys,tree_amount)
        self.trees=[DecisionTree(subXs[i],subys[i],max_depth,prune_data_threshold) for i in range(len(subXs))]
    
    def classify(self, xs: np.array):
        if len(xs.shape) == 2:
            return np.array([self.classify(x) for x in xs])
        else:
            results = [tree.classify(xs) for tree in self.trees]
            options,counts=np.unique(results,return_counts=True)
            best= None
            bestCount=0
            for i in range(len(options)):
                if counts[i]>bestCount:
                    bestCount=counts[i]
                    best = options[i]
            return best
    
    def score(self,Xs,ys):
        assert len(Xs) == len(ys) , "The input was not of the same size"
        _, counts = np.unique(self.classify(Xs) == ys,return_counts=True)
        if len(counts) == 1:
            return 0
        return counts[1]/len(Xs)

    @staticmethod
    def createSubsets(array: np.array,array2: np.array ,num):
        assert len(array)==len(array2), "The size of the two input arrays should be the same"
        indecies = np.arange(len(array))
        index_list = [ np.random.choice(indecies,size=len(array)//3) for i in range(num)]
        return [array[i] for i in index_list], [array2[i] for i in index_list]
  
    def __str__(self):
        return_string =""
        for tree in self.trees:
            return_string += str(tree) + "\n"
        return return_string

