import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Initialize random state
rn = 42
np.random.seed(rn)

lbP = [-3, -3, -20, -3, -3, -4] 
ubP = [3, 3, 20, 3, 3, 2]
lbE = [-3, -3, -20, -3, -3, -4]
ubE = [3, 3, 20, 3, 3, 2]

file_sets = {
    "normal": [f"Data_normal_{i}.csv" for i in range(1, 7)],
    "highPE": [f"Data_highPE_{i}.csv" for i in range(1, 7)],
    "outlier": [f"Data_outlier_{i}.csv" for i in range(1, 7)]
}

# Dictionary to store models and data
models = {}

# Dictionary to store lb_out, ub_out, and inc
cbar_vals = {}

def build_and_store_models(file_set_name, file_names):
    models[file_set_name] = {}
    cbar_vals[file_set_name] = {}  # Ensure each dataset type has its own sub-dictionary

    for i, file_name in enumerate(file_names):
        data = pd.read_csv(file_name)
        X = data[['P', 'E']]
        y = data['Z']
        
        poly_features = pd.DataFrame({
            'P': data['P'],
            'E': data['E'],
            'P^2': data['P'] ** 2,
            'E^2': data['E'] ** 2,
            'P*E': data['P'] * data['E']
        })
        
        poly_model = LinearRegression().fit(poly_features, y)
        tree_model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=1, random_state=rn).fit(X, y)
        
        models[file_set_name][i] = {
            "data": data,
            "poly_model": poly_model,
            "tree_model": tree_model
        }
        
        # Store colorbar values only for "normal" dataset
        lb_out = np.round(data["Z"].min(), 1)
        ub_out = np.round(data["Z"].max(), 1)
        inc = (ub_out - lb_out) / 9
        cbar_vals[file_set_name][i] = (lb_out, ub_out, inc)

# Build models once for each dataset type
for key, files in file_sets.items():
    build_and_store_models(key, files)

# Function to plot models
def plot_model(ax, model, X, y, lbx, ubx, lby, uby, lb_out, ub_out, inc, colormap='gray', scatter=True):
    x_vals = np.linspace(lbx, ubx, 500)
    y_vals = np.linspace(lby, uby, 500)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    X_input = pd.DataFrame({'P': X_grid.ravel(), 'E': Y_grid.ravel()})
    X_input['P^2'] = X_input['P'] ** 2
    X_input['E^2'] = X_input['E'] ** 2
    X_input['P*E'] = X_input['P'] * X_input['E']
    
    if isinstance(model, DecisionTreeRegressor):
        Z = model.predict(X_input[['P', 'E']])
    else:
        Z = model.predict(X_input)
    
    Z = Z.reshape(X_grid.shape)
    im = ax.contourf(X_grid, Y_grid, Z, levels=np.arange(lb_out, ub_out+inc, inc), cmap=colormap, extend='both', alpha=0.8)
    
    if scatter:
        sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap, vmin=lb_out, vmax=ub_out+inc, edgecolor='w', linewidths=0.5, s=20)
    
    cbar = plt.colorbar(im, ax=ax, location='top', shrink=0.8, ticks=np.arange(lb_out, ub_out+inc, inc), format="%.1f")
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Outcome', rotation=0, loc='center')
    
    ax.set_xlabel('Person')
    ax.set_ylabel('Environment')
    ax.set_aspect('equal', adjustable='box')
    return im

# Regular plots
fig, axes = plt.subplots(2, 6, figsize=(25, 10))
for i in range(6):
    data = models['normal'][i]['data']
    poly_model = models['normal'][i]['poly_model']
    tree_model = models['normal'][i]['tree_model']
    lb_out, ub_out, inc = cbar_vals['normal'][i]
    
    plot_model(axes[0, i], poly_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc)
    plot_model(axes[1, i], tree_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc)

fig.tight_layout()
plt.show()

# Outlier plots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
data = models['normal'][0]['data']
poly_model = models['normal'][0]['poly_model']
tree_model = models['normal'][0]['tree_model']
lb_out, ub_out, inc = cbar_vals['normal'][0]
plot_model(axes[0, 0], poly_model, data[['P', 'E']].values, data['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc)
plot_model(axes[0, 1], tree_model, data[['P', 'E']].values, data['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc)
data_outlier = models['outlier'][0]['data']
plot_model(axes[1, 0], models['outlier'][0]['poly_model'], data_outlier[['P', 'E']].values, data_outlier['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc)
plot_model(axes[1, 1], models['outlier'][0]['tree_model'], data_outlier[['P', 'E']].values, data_outlier['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc)
plt.show()

# Concentrated plots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
data = models['normal'][1]['data']
poly_model = models['normal'][1]['poly_model']
tree_model = models['normal'][1]['tree_model']
lb_out, ub_out, inc = cbar_vals['normal'][1]
plot_model(axes[0, 0], poly_model, data[['P', 'E']].values, data['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc)
plot_model(axes[0, 1], tree_model, data[['P', 'E']].values, data['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc)
data_highPE = models['highPE'][1]['data']
plot_model(axes[1, 0], models['highPE'][1]['poly_model'], data_highPE[['P', 'E']].values, data_highPE['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc)
plot_model(axes[1, 1], models['highPE'][1]['tree_model'], data_highPE[['P', 'E']].values, data_highPE['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc)
plt.show()

# Appendix outlier plots
fig, axes = plt.subplots(2, 6, figsize=(25, 10))
for i in range(6):
    data = models['outlier'][i]['data']
    poly_model = models['outlier'][i]['poly_model']
    tree_model = models['outlier'][i]['tree_model']
    lb_out, ub_out, inc = cbar_vals['normal'][i]
    
    plot_model(axes[0, i], poly_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc)
    plot_model(axes[1, i], tree_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc)

fig.tight_layout()
plt.show()

# Appendix concentrated plots
fig, axes = plt.subplots(2, 6, figsize=(25, 10))
for i in range(6):
    data = models['highPE'][i]['data']
    poly_model = models['highPE'][i]['poly_model']
    tree_model = models['highPE'][i]['tree_model']
    lb_out, ub_out, inc = cbar_vals['normal'][i]
    
    plot_model(axes[0, i], poly_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc)
    plot_model(axes[1, i], tree_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc)

fig.tight_layout()
plt.show()
