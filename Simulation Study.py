#Installing and importing the required libraries and modules
!pip install pandas numpy scikit-learn matplotlib
import pandas as pd
import numpy as np
import random
from sklearn.utils import check_random_state
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

#Setting random seeds
rn=42
random.seed(rn)
np.random.seed(rn)
random_state=check_random_state(rn)

#Define the combinations of coefficients
coefficients=[
    (0.214, -0.222, -0.16, 0.051, 0.111),
    (-0.011, 0.28, 0.041, -0.127, 0.082),
    (0.019, 0.262, -0.012, -0.022, 0.056),
    (-0.09, 0.11, -0.13, -0.17, 0.23),
    (-0.06, 0.11, -0.05, 0.01, 0.11),
    (-0.11, 0.25, -0.01, 0, 0)]

#Setting the bounds for P and values and defining the pattern names
lbP=[-3, -3, -20, -3, -3, -4] 
ubP=[3, 3, 20, 3, 3, 2]
lbE=[-3, -3, -20, -3, -3, -4]
ubE=[3, 3, 20, 3, 3, 2]

patterns=['Stable Fit', 'Elevated Fit', 'Extreme Fit', 'Symmetrical Misfit', 'Asymmetrical Misfit', 'Beneficial Misfit']

#Function to round values to nearest 0.5
def round_it(x, y):
    return np.round(x/y)*y

#Dictionary to store datasets
datasets={"normal":[], "highPE":[], "outlier":[]}

#Create a DataFrame to store all datasets
for idx, (a, b, c, d, e) in enumerate(coefficients):
    #Default
    n=300
    
    P=lbP[idx]+np.random.rand(n)*(ubP[idx]-lbP[idx])
    E=lbE[idx]+np.random.rand(n)*(ubE[idx]-lbE[idx])
    
    P=round_it(P, (ubP[idx]-lbP[idx])/30)
    E=round_it(E, (ubP[idx]-lbP[idx])/30)
    
    Z=a*P+b*E+c*P**2+d*E**2+e*P*E
    error_std = 0.05 * abs(Z.max() - Z.min())
    noise = np.random.normal(loc=0, scale=error_std, size=n)
    Z = Z + noise
    df=pd.DataFrame({'P':P, 'E':E, 'Z':Z})
    datasets["normal"].append(df)
    
    #Outlier
    if idx==0:
        outlier1={'P': 0.2, 'E': 1.2, 'Z': -3}
        outlier2={'P': 0.2, 'E': 0.8, 'Z': -3.2}
        outlier3={'P': 0, 'E': 1.4, 'Z': -2.8}
        outlier4={'P': 0, 'E': 1, 'Z': -3}
        outlier5={'P': 0, 'E': 0.8, 'Z': -3.2}
        outlier6={'P': -0.2, 'E': 1.6, 'Z': -3}
        outlier7={'P': -0.2, 'E': 1.2, 'Z': -2.8}
        outlier8={'P': -0.2, 'E': 0.8, 'Z': -3}
        outlier9={'P': -0.4, 'E': 1, 'Z': -2.6}
        outlier10={'P': -0.4, 'E': 0.6, 'Z': -3.2}
    elif idx==1:
        outlier1={'P': 1.8, 'E': 2.4, 'Z': -2}
        outlier2={'P': 1.8, 'E': 2.2, 'Z': -1.8}
        outlier3={'P': 1.8, 'E': 1.8, 'Z': -2}
        outlier4={'P': 2, 'E': 2.4, 'Z': -1.6}
        outlier5={'P': 2, 'E': 2.2, 'Z': -1.8}
        outlier6={'P': 2, 'E': 2, 'Z': -2}
        outlier7={'P': 2, 'E': 1.6, 'Z': -1.6}
        outlier8={'P': 2.2, 'E': 2.6, 'Z': -2}
        outlier9={'P': 2.2, 'E': 1.8, 'Z': -1.8}
        outlier10={'P': 2.2, 'E': 1.4, 'Z': -2}
    elif idx==2:
        outlier1={'P': 0, 'E': 32/3, 'Z': -32}
        outlier2={'P': 4/3, 'E': 32/3, 'Z': -34}
        outlier3={'P': 8/3, 'E': 12, 'Z': -30}
        outlier4={'P': 8/3, 'E': 32/3, 'Z': -32}
        outlier5={'P': 8/3, 'E': 28/3, 'Z': -30}
        outlier6={'P': 4, 'E': 32/3, 'Z': -34}
        outlier7={'P': 16/3, 'E': 28/3, 'Z': -30}
        outlier8={'P': 16/3, 'E': 12, 'Z': -30}
        outlier9={'P': 20/3, 'E': 32/3, 'Z': -32}
        outlier10={'P': 8, 'E': 12, 'Z': -34}
    elif idx==3:
        outlier1={'P': -1, 'E': -2.2, 'Z': -4.2}
        outlier2={'P': -1, 'E': -2.6, 'Z': -4.4}
        outlier3={'P': -1.2, 'E': -2.2, 'Z': -4}
        outlier4={'P': -1.2, 'E': -2.4, 'Z': -4.2}
        outlier5={'P': -1.2, 'E': -2.8, 'Z': -4}
        outlier6={'P': -1.2, 'E': -3, 'Z': -4.4}
        outlier7={'P': -1.4, 'E': -2, 'Z': -4.2}
        outlier8={'P': -1.4, 'E': -2.2, 'Z': -4.2}
        outlier9={'P': -1.4, 'E': -2.6, 'Z': -4}
        outlier10={'P': -1.4, 'E': -3, 'Z': -4.4}
    elif idx==4:
        outlier1={'P': 0.4, 'E': 2.2, 'Z': -1.6}
        outlier2={'P': 0.4, 'E': 1.8, 'Z': -1.4}
        outlier3={'P': 0.4, 'E': 1.6, 'Z': -1.2}
        outlier4={'P': 0.4, 'E': 1.2, 'Z': -1.6}
        outlier5={'P': 0.6, 'E': 2.4, 'Z': -1.6}
        outlier6={'P': 0.6, 'E': 2.2, 'Z': -1.2}
        outlier7={'P': 0.6, 'E': 1.8, 'Z': -1.2}
        outlier8={'P': 0.6, 'E': 1.4, 'Z': -1.4}
        outlier9={'P': 0.6, 'E': 1.2, 'Z': -1.6}
        outlier10={'P': 0.6, 'E': 0.8, 'Z': -1.4}
    elif idx==5:
        outlier1={'P': 0, 'E': -3, 'Z': 0.6}
        outlier2={'P': 0, 'E': -3.2, 'Z': 0.4}
        outlier3={'P': -0.2, 'E': -2.8, 'Z': 0.8}
        outlier4={'P': -0.2, 'E': -3.2, 'Z': 0.4}
        outlier5={'P': -0.2, 'E': -3.4, 'Z': 0.4}
        outlier6={'P': -0.4, 'E': -3, 'Z': 0.8}
        outlier7={'P': -0.4, 'E': -3.6, 'Z': 0.6}
        outlier8={'P': -0.4, 'E': -3.8, 'Z': 0.8}
        outlier9={'P': -0.4, 'E': -4, 'Z': 0.4}
        outlier10={'P': -0.6, 'E': -3, 'Z': 0.6}

    outliers_df=pd.DataFrame([outlier1, outlier2, outlier3, outlier4, outlier5, outlier6, outlier7, outlier8, outlier9, outlier10])
    df_outlier=pd.concat([df, outliers_df], ignore_index=True)
    datasets["outlier"].append(df_outlier)

    #Concentrated on high values of P and E
    P_highPE=lbP[idx]+np.random.beta(3, 2, size=n)*(ubP[idx]-lbP[idx])
    E_highPE=lbE[idx]+np.random.beta(3, 2, size=n)*(ubE[idx]-lbE[idx])
    
    P_highPE=round_it(P_highPE, (ubP[idx]-lbP[idx])/30)
    E_highPE=round_it(E_highPE, (ubP[idx]-lbP[idx])/30)
    
    Z_highPE=a*P_highPE+b*E_highPE+c*P_highPE**2+d*E_highPE**2+e*P_highPE*E_highPE
    error_std = 0.05 * abs(Z_highPE.max() - Z_highPE.min())
    noise = np.random.normal(loc=0, scale=error_std, size=n)
    Z_highPE = Z_highPE + noise
    df_highPE = pd.DataFrame({'P':P_highPE, 'E':E_highPE, 'Z':Z_highPE})
    datasets["highPE"].append(df_highPE)

#Plotting Figure 3
title_font = 26
label_font = 20
tick_font = 16
left_title_font = 26
title_pad = 10

fig, axes = plt.subplots(2, 6, figsize=(24, 8), sharey='row')
depths = list(range(1, 11))
min_samples = list(range(1, 11))

for idx, df in enumerate(datasets["normal"]):
    X = df[["P", "E"]]
    y = df["Z"]
    r2_scores_depth = []
    for depth in depths:
        model = DecisionTreeRegressor(max_depth=depth, random_state=rn)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2_scores_depth.append(r2_score(y, y_pred))
    ax1 = axes[0, idx]
    ax1.plot(depths, r2_scores_depth, marker='o', color='black')
    ax1.set_title(f'{patterns[idx]}', fontsize=title_font, fontweight='bold', pad=title_pad)
    ax1.set_xlabel('maxDepth', fontsize=label_font)
    if idx == 0:
        ax1.set_ylabel('R²', fontsize=label_font)
    ax1.set_xticks(depths)
    ax1.tick_params(axis='both', labelsize=tick_font)
    ax1.grid(True)
    if idx == 0:
        ax1.text(-0.3, 0.5, 'maxDepth', ha='center', va='center', rotation=90,
                 fontsize=left_title_font, fontweight='bold', transform=ax1.transAxes)
    r2_scores_leaf = []
    for m in min_samples:
        model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=m, random_state=rn)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2_scores_leaf.append(r2_score(y, y_pred))
    ax2 = axes[1, idx]
    ax2.plot(min_samples, r2_scores_leaf, marker='o', color='black')
    ax2.set_xlabel('minSamples', fontsize=label_font)
    if idx == 0:
        ax2.set_ylabel('R²', fontsize=label_font)
    ax2.set_xticks(min_samples)
    ax2.tick_params(axis='both', labelsize=tick_font)
    ax2.grid(True)
    if idx == 0:
        ax2.text(-0.3, 0.5, 'minSamples', ha='center', va='center', rotation=90,
                 fontsize=left_title_font, fontweight='bold', transform=ax2.transAxes)

plt.tight_layout()
plt.show()

#Dictionary to store models, R-squares, lb_out, ub_out, and inc
models={}
r2_vals={}
cbar_vals={}

def build_and_store_models(dataset_type, dataset):
    models[dataset_type]={}
    r2_vals[dataset_type]={}
    cbar_vals[dataset_type]={}

    for i in range(len(files)):
        data=dataset[i]
        X=data[['P', 'E']]
        y=data['Z']
        
        poly_features=pd.DataFrame({
            'P':data['P'],
            'E':data['E'],
            'P^2':data['P']**2,
            'E^2':data['E']**2,
            'P*E':data['P']*data['E']})
        
        poly_model=LinearRegression().fit(poly_features, y)
        tree_model=DecisionTreeRegressor(max_depth=6, min_samples_leaf=3, random_state=rn).fit(X, y)
        
        models[dataset_type][i]={
            "data":data,
            "poly_model":poly_model,
            "tree_model":tree_model}
        
        pred_poly=poly_model.predict(poly_features)
        pred_tree=tree_model.predict(X)

        r2_vals[dataset_type][i]={
            "poly_model":r2_score(y, pred_poly),
            "tree_model":r2_score(y, pred_tree)}
        
        lb_out=np.round(data["Z"].min(), 1)
        ub_out=np.round(data["Z"].max(), 1)
        inc=(ub_out-lb_out)/9
        cbar_vals[dataset_type][i]=(lb_out, ub_out, inc)

#Build models once for each dataset type
for key, files in datasets.items():
    build_and_store_models(key, files)

#Function to plot models
def plot_model(ax, model, X, y, lbx, ubx, lby, uby, lb_out, ub_out, inc, colormap='gray', scatter=True, title=None, title_font=26, title_pad=70, left_title=None, left_title_font=26, r2=None, r2_font=18, r2_x=0.5, r2_y=-0.22, label_font=16, left_x=-0.25, left_y=0.5, tick_font=16, clb_tick_font=8):
    x_vals=np.linspace(lbx, ubx, 500)
    y_vals=np.linspace(lby, uby, 500)
    X_grid, Y_grid=np.meshgrid(x_vals, y_vals)
    X_input=pd.DataFrame({'P': X_grid.ravel(), 'E': Y_grid.ravel()})
    X_input['P^2']=X_input['P']**2
    X_input['E^2']=X_input['E']**2
    X_input['P*E']=X_input['P']*X_input['E']
    
    if isinstance(model, DecisionTreeRegressor):
        Z=model.predict(X_input[['P', 'E']])
    else:
        Z=model.predict(X_input)
    
    Z=Z.reshape(X_grid.shape)
    im=ax.contourf(X_grid, Y_grid, Z, levels=np.arange(lb_out, ub_out+inc, inc), cmap=colormap, extend='both', alpha=0.8)
    
    if scatter:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap, vmin=lb_out, vmax=ub_out+inc, edgecolor='w', linewidths=0.5, s=20)
    
    cbar=plt.colorbar(im, ax=ax, location='top', shrink=0.8, ticks=np.arange(lb_out, ub_out+inc, inc), format="%.1f")
    cbar.ax.tick_params(labelsize=clb_tick_font)
    cbar.set_label('Outcome', rotation=0, loc='center', fontsize=label_font)
    
    ax.set_xlabel('Person', fontsize=label_font)
    ax.set_ylabel('Environment', fontsize=label_font)
    ax.tick_params(axis='both', labelsize=tick_font)
    ax.set_aspect('equal', adjustable='box')

    if title:
        ax.set_title(title, fontsize=title_font, fontweight='bold', pad=title_pad)

    if left_title:
        ax.text(left_x, left_y, left_title, ha='center', va='center', rotation=90, fontsize=left_title_font, fontweight='bold', transform=ax.transAxes)

    if r2 is not None:
        ax.text(r2_x, r2_y, f'R²={r2:.2f}', ha='center', va='center', fontsize=r2_font, transform=ax.transAxes)
    return im

#Plotting Figure 4
fig, axes=plt.subplots(2, 6, figsize=(25, 11))
for i in range(6):
    data=models['normal'][i]['data']
    poly_model=models['normal'][i]['poly_model']
    tree_model=models['normal'][i]['tree_model']
    poly_r2=r2_vals['normal'][i]['poly_model']
    tree_r2=r2_vals['normal'][i]['tree_model']
    lb_out, ub_out, inc=cbar_vals['normal'][i]
    if i==0:
        plot_model(axes[0, i], poly_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, title=patterns[i], left_title='PRA', r2=poly_r2)
        plot_model(axes[1, i], tree_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, left_title='DT', r2=tree_r2)
    else:
        plot_model(axes[0, i], poly_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, title=patterns[i], r2=poly_r2)
        plot_model(axes[1, i], tree_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, r2=tree_r2)

fig.tight_layout()
plt.show()

#Plotting Figure 5
fig, axes=plt.subplots(2, 2, figsize=(10, 11))
data=models['normal'][0]['data']
lb_out, ub_out, inc=cbar_vals['normal'][0]
plot_model(axes[0, 0], models['normal'][0]['poly_model'], data[['P', 'E']].values, data['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc, title='PRA', left_title='Without Outliers', r2=r2_vals['normal'][0]['poly_model'], title_font=22, title_pad=60, left_title_font=22, r2_font=14, r2_x=0.5, r2_y=-0.18, label_font=12, left_x=-0.25, left_y=0.5, tick_font=12, clb_tick_font=10)
plot_model(axes[0, 1], models['normal'][0]['tree_model'], data[['P', 'E']].values, data['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc, title='DT', r2=r2_vals['normal'][0]['tree_model'], title_font=22, title_pad=60, left_title_font=22, r2_font=14, r2_x=0.5, r2_y=-0.18, label_font=12, left_x=-0.25, left_y=0.5, tick_font=12, clb_tick_font=10)
data_outlier=models['outlier'][0]['data']
plot_model(axes[1, 0], models['outlier'][0]['poly_model'], data_outlier[['P', 'E']].values, data_outlier['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc, left_title='With Outliers', r2=r2_vals['outlier'][0]['poly_model'], title_font=22, title_pad=60, left_title_font=22, r2_font=14, r2_x=0.5, r2_y=-0.18, label_font=12, left_x=-0.25, left_y=0.5, tick_font=12, clb_tick_font=10)
plot_model(axes[1, 1], models['outlier'][0]['tree_model'], data_outlier[['P', 'E']].values, data_outlier['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc, r2=r2_vals['outlier'][0]['tree_model'], title_font=22, title_pad=60, left_title_font=22, r2_font=14, r2_x=0.5, r2_y=-0.18, label_font=12, left_x=-0.25, left_y=0.5, tick_font=12, clb_tick_font=10)
plt.show()

#Plotting Figure 6
fig, axes=plt.subplots(2, 2, figsize=(10, 11))
data=models['normal'][1]['data']
lb_out, ub_out, inc=cbar_vals['normal'][1]
plot_model(axes[0, 0], models['normal'][1]['poly_model'], data[['P', 'E']].values, data['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc, title='PRA', left_title='Uniform Distribution', r2=r2_vals['normal'][1]['poly_model'], title_font=22, title_pad=60, left_title_font=22, r2_font=14, r2_x=0.5, r2_y=-0.18, label_font=12, left_x=-0.25, left_y=0.5, tick_font=12, clb_tick_font=10)
plot_model(axes[0, 1], models['normal'][1]['tree_model'], data[['P', 'E']].values, data['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc, title='DT', r2=r2_vals['normal'][1]['tree_model'], title_font=22, title_pad=60, left_title_font=22, r2_font=14, r2_x=0.5, r2_y=-0.18, label_font=12, left_x=-0.25, left_y=0.5, tick_font=12, clb_tick_font=10)
data_highPE=models['highPE'][1]['data']
plot_model(axes[1, 0], models['highPE'][1]['poly_model'], data_highPE[['P', 'E']].values, data_highPE['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc, left_title='Skewed Distribution', r2=r2_vals['highPE'][1]['poly_model'], title_font=22, title_pad=60, left_title_font=22, r2_font=14, r2_x=0.5, r2_y=-0.18, label_font=12, left_x=-0.25, left_y=0.5, tick_font=12, clb_tick_font=10)
plot_model(axes[1, 1], models['highPE'][1]['tree_model'], data_highPE[['P', 'E']].values, data_highPE['Z'].values, -3, 3, -3, 3, lb_out, ub_out, inc, r2=r2_vals['highPE'][1]['tree_model'], title_font=22, title_pad=60, left_title_font=22, r2_font=14, r2_x=0.5, r2_y=-0.18, label_font=12, left_x=-0.25, left_y=0.5, tick_font=12, clb_tick_font=10)
plt.show()

# Plotting all DT models fitted to the n=300 and n=100 datasets
fig, axes = plt.subplots(2, 6, figsize=(25, 11))

for i in range(6):
    data_full = models['normal'][i]['data'].copy()
    tree_model_orig = models['normal'][i]['tree_model']
    lb_out, ub_out, inc=cbar_vals['normal'][i]

    if i == 0:
        plot_model(axes[0, i],
                   tree_model_orig,
                   data_full[['P', 'E']].values,
                   data_full['Z'].values,
                   lbP[i], ubP[i],
                   lbE[i], ubE[i],
                   lb_out, ub_out, inc,
                   title=patterns[i],
                   left_title='n=300',
                   r2=r2_vals['normal'][i]['tree_model'])
    else:
        plot_model(axes[0, i],
                   tree_model_orig,
                   data_full[['P', 'E']].values,
                   data_full['Z'].values,
                   lbP[i], ubP[i],
                   lbE[i], ubE[i],
                   lb_out, ub_out, inc,
                   title=patterns[i],
                   r2=r2_vals['normal'][i]['tree_model'])

    data_subset = data_full.sample(n=100, random_state=42).reset_index(drop=True)
    X_subset = data_subset[['P', 'E']]
    y_subset = data_subset['Z']
    tree_model_subset = DecisionTreeRegressor(max_depth=6, min_samples_leaf=3)
    tree_model_subset.fit(X_subset, y_subset)

    if i == 0:
        plot_model(axes[1, i],
                   tree_model_subset,
                   X_subset.values,
                   y_subset.values,
                   lbP[i], ubP[i],
                   lbE[i], ubE[i],
                   lb_out, ub_out, inc,
                   left_title='n=100',
                   r2=r2_score(y_subset, tree_model_subset.predict(X_subset)))
    else:
        plot_model(axes[1, i],
                   tree_model_subset,
                   X_subset.values,
                   y_subset.values,
                   lbP[i], ubP[i],
                   lbE[i], ubE[i],
                   lb_out, ub_out, inc,
                   r2=r2_score(y_subset, tree_model_subset.predict(X_subset)))

plt.tight_layout()
plt.show()

#Plotting all PRA and DT models fitted to the datasets with outliers
fig, axes=plt.subplots(2, 6, figsize=(25, 11))
for i in range(6):
    data=models['outlier'][i]['data']
    poly_model=models['outlier'][i]['poly_model']
    tree_model=models['outlier'][i]['tree_model']
    poly_r2=r2_vals['outlier'][i]['poly_model']
    tree_r2=r2_vals['outlier'][i]['tree_model']
    lb_out, ub_out, inc=cbar_vals['normal'][i]
    if i==0:
        plot_model(axes[0, i], poly_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, title=patterns[i], left_title='PRA', r2=poly_r2)
        plot_model(axes[1, i], tree_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, left_title='DT', r2=tree_r2)
    else:
        plot_model(axes[0, i], poly_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, title=patterns[i], r2=poly_r2)
        plot_model(axes[1, i], tree_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, r2=tree_r2)

fig.tight_layout()
plt.show()

#Plotting all PRA and DT models fitted to the datasets with skewed P and E values
fig, axes=plt.subplots(2, 6, figsize=(25, 11))
for i in range(6):
    data=models['highPE'][i]['data']
    poly_model=models['highPE'][i]['poly_model']
    tree_model=models['highPE'][i]['tree_model']
    poly_r2=r2_vals['highPE'][i]['poly_model']
    tree_r2=r2_vals['highPE'][i]['tree_model']
    lb_out, ub_out, inc=cbar_vals['normal'][i]
    if i==0:
        plot_model(axes[0, i], poly_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, title=patterns[i], left_title='PRA', r2=poly_r2)
        plot_model(axes[1, i], tree_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, left_title='DT', r2=tree_r2)
    else:
        plot_model(axes[0, i], poly_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, title=patterns[i], r2=poly_r2)
        plot_model(axes[1, i], tree_model, data[['P', 'E']].values, data['Z'].values, lbP[i], ubP[i], lbE[i], ubE[i], lb_out, ub_out, inc, r2=tree_r2)

fig.tight_layout()
plt.show()