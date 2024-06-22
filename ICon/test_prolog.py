from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.datasets import make_regression

# Generate a sample dataset
X, y = make_regression(n_samples=5, n_features=2, noise=0.1)

# Train the decision tree regressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

tree = regressor.tree_
# Extract the features and thresholds
features = tree.feature
thresholds = tree.threshold
left_children = tree.children_left
right_children = tree.children_right
values = tree.value


def extract_tree_structure(tree, feature_names):
    tree_ = tree.tree_
    feature_names = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[node]
            threshold = tree_.threshold[node]
            print(f"{indent}if ({name} <= {threshold:}) {{")
            recurse(tree_.children_left[node], depth + 1)
            print(f"{indent}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            print(f"{indent}}}")
        else:
            print(f"{indent}return {tree_.value[node]}")

    recurse(0, 0)


def tree_to_prolog(regressor, node_id=0, parent_id=None):
    # Check if it's a leaf node
    if regressor.tree_.children_left[node_id] == -1:
        # Leaf node
        value = regressor.tree_.value[node_id][0, 0]  # Assuming single target value
        print(f"leaf({node_id}, {value}).")
    else:
        # Decision node
        feature = regressor.tree_.feature[node_id]
        threshold = regressor.tree_.threshold[node_id]
        left_child = regressor.tree_.children_left[node_id]
        right_child = regressor.tree_.children_right[node_id]
        print(f"node({node_id}, {feature}, {threshold}, {left_child}, {right_child}, _).")
        # Recursively print left and right children
        tree_to_prolog(regressor, left_child, node_id)
        tree_to_prolog(regressor, right_child, node_id)


extract_tree_structure(regressor, ['feature1', 'feature2'])
tree_to_prolog(regressor)

#predict value for 2,3
pred = regressor.predict([[2,3]])
