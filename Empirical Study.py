#Installing and importing the required libraries and modules
#!pip install pandas numpy scikit-learn dtreeviz matplotlib seaborn
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
import matplotlib.patches as mpatches
import random
from sklearn.utils import check_random_state

#Setting random seeds
rn=42
random.seed(rn)
np.random.seed(rn)
random_state=check_random_state(rn)

#Importing the data
data=pd.read_csv("Data.csv")
for i, col in enumerate(data.columns):
    globals()[f'var{i+1}']=col

#Please enter a unique label for each variable in your data
labels=['P','J','T','S']

#Please enter the lower and upper bound for P (lbx and ubx) and E (lby and uby) values 
lbx=-2
ubx=2
lby=-2
uby=2

#Preparing the data for polynomial regression analysis
regr_data=pd.DataFrame()
regr_data[labels[0]]=data.iloc[:, 0]
regr_data[labels[1]]=data.iloc[:, 1]
regr_data[labels[0]+'^2']=regr_data[labels[0]]**2
regr_data[labels[1]+'^2']=regr_data[labels[1]]**2
regr_data[labels[0]+'*'+labels[1]]=regr_data[labels[0]]*regr_data[labels[1]]
for col_name, col_data in data.iloc[:, 2:].items():
    regr_data[col_name]=col_data

#Building decision tree and polynomial regression models
def build_decision_tree_models(X, y, depth, min_leaf):
    r2=[]
    clf = DecisionTreeRegressor(min_samples_leaf=min_leaf, max_depth=depth, random_state=rn)
    clf.fit(X, y)
    prediction=clf.predict(X)
    model=clf
    r2=r2_score(y, prediction)
    return model, prediction, r2

DTmodel, DTpred, DTr2 =build_decision_tree_models(data.iloc[:,0:2], data.iloc[:,-1], depth=4, min_leaf=5)
DTmodel_cont, DTpred_cont, DTr2_cont =build_decision_tree_models(data.iloc[:,0:-1], data.iloc[:,-1], depth=4, min_leaf=5)

def build_polynomial_regression_model(X, y):
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    prediction=regr.predict(X)
    r2=r2_score(prediction, y)
    return regr, prediction, r2

Rmodel, Rpred, Rr2 =build_polynomial_regression_model(regr_data.iloc[:,0:5], regr_data.iloc[:,-1])

#Plotter
def plotter(model, data, var1, var2, labels, lbx, ubx, lby, uby, lb_out, ub_out, inc_levels, inc_counters, legend_title, colormap, x_axis, y_axis, scat, jitter, ax):
    x=np.linspace(lbx, ubx, 500)
    y=np.linspace(lby, uby, 500)
    X, Y = np.meshgrid(x, y)
    x=X.flatten()
    y=Y.flatten()
    if isinstance(model, DecisionTreeRegressor):
        df=pd.DataFrame({var1:x, var2:y})
    else:
        x2=x**2
        y2=y**2
        xy=x*y
        df=pd.DataFrame({labels[0]:x, labels[1]:y, labels[0]+'^2':x2, labels[1]+'^2':y2, labels[0]+'*'+labels[1]:xy})
    Z=model.predict(df)
    c=Z.reshape(500,500)
    if jitter:
        data_jit=pd.DataFrame()
        data_jit[labels[0]+'_jittered']=data.iloc[:,0]+np.random.randn(len(data.iloc[:,0]))*0.002*(ubx-lbx)
        data_jit[labels[0]+'_jittered'][data_jit[labels[0]+'_jittered']>=ubx]=ubx-abs(np.random.randn()*0.002*(ubx-lbx))
        data_jit[labels[0]+'_jittered'][data_jit[labels[0]+'_jittered']<=lbx]=lbx+abs(np.random.randn()*0.002*(ubx-lbx))
        data_jit[labels[1]+'_jittered']=data.iloc[:,1]+np.random.randn(len(data.iloc[:,1]))*0.002*(uby-lby)
        data_jit[labels[1]+'_jittered'][data_jit[labels[1]+'_jittered']>=uby]=uby-abs(np.random.randn()*0.002*(uby-lby))
        data_jit[labels[1]+'_jittered'][data_jit[labels[1]+'_jittered']<=lby]=lby+abs(np.random.randn()*0.002*(uby-lby))
    if ax is None:
        if isinstance(model, DecisionTreeRegressor):
            im=plt.imshow(Z.reshape(500,500), extent=[lbx, ubx, lby, uby], vmin=lb_out, vmax=ub_out, origin='lower',
           cmap=colormap,aspect='1')
        else:
            plt.contourf(X, Y, c, levels=np.arange(lb_out, ub_out+0.00001, inc_counters), cmap=colormap, extend='both')
        cbar=plt.colorbar(location='top', shrink=0.6, ticks=np.arange(lb_out, ub_out+0.00001, inc_levels))
        cbar.set_label(legend_title, rotation=0, loc='center')
        plt.xlim(lbx,ubx)
        plt.ylim(lby,uby)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        if isinstance(model, DecisionTreeRegressor):
            for t in cbar.ax.get_xticklabels():
                t.set_fontsize(8)
        else:
            ax=plt.gca()
            ax.set_aspect('equal', adjustable='box')
            cbar.ax.tick_params(labelsize=8)
        if scat:
            if jitter:
                plt.scatter(data_jit[labels[0]+'_jittered'],data_jit[labels[1]+'_jittered'],c=data.iloc[:,-1], cmap=colormap, vmin=lb_out, vmax=ub_out, edgecolors='w', linewidths=0.5, s=20)
            else:
                plt.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,-1], cmap=colormap, vmin=lb_out, vmax=ub_out, edgecolors='w', linewidths=0.5, s=20)
        plt.show()
        plt.close()
    else:
        if isinstance(model, DecisionTreeRegressor):
            im=ax.imshow(Z.reshape(500,500), extent=[lbx, ubx, lby, uby], vmin=lb_out, vmax=ub_out, origin='lower',
           cmap=colormap,aspect='1')
            values=np.unique(Z)
            colors=[ im.cmap(im.norm(value)) for value in values]
            patches=[mpatches.Patch(color=colors[i], label="{l}".format(l=round(values[i],2)) ) for i in range(len(values))]
        else:
            ax.contourf(X, Y, c, levels=np.arange(lb_out, ub_out+0.00001, inc_counters), cmap=colormap, extend='both')
        if scat:
            if jitter:
                ax.scatter(data_jit[labels[0]+'_jittered'],data_jit[labels[1]+'_jittered'],c=data.iloc[:,-1], cmap=colormap, vmin=lb_out, vmax=ub_out, edgecolors='w', linewidths=0.5, s=10)
            else:
                ax.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,-1], cmap=colormap, vmin=lb_out, vmax=ub_out, edgecolors='w', linewidths=0.5, s=10)
        ax.set_xlim(lbx,ubx)
        ax.set_ylim(lby,uby)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis, labelpad=0)
        if isinstance(model, DecisionTreeRegressor):
            return im

#Plotting Figure 1
fig = plt.figure(figsize=(80,40))
_ = plot_tree(DTmodel, feature_names=labels[0:2], filled=True)
plt.show()
plt.close(fig)

#Please enter the depth level of the decision tree model, legend of the title, axis names, axis bounds, increment level on legend, whether having jittered points, and the colorbar to be appeared in the counter plot
legend_title='Job Satisfaction'
x_axis='Person'
y_axis='Job'
lb_out=3.5
ub_out=5
inc=0.25
colormap='bwr'
scat=True
jitter=True

#Plotting Figure 2
plotter(DTmodel, data, var1, var2, labels, lbx, ubx, lby, uby, lb_out, ub_out, inc, inc, legend_title, colormap, x_axis, y_axis, scat, jitter, None)

#Plotting Figure 6
fig = plt.figure(figsize=(80,40))
_ = plot_tree(DTmodel_cont, feature_names=labels[0:-1], filled=True)
plt.show()
plt.close(fig)

# Importing the clusters data
df_cl = pd.read_csv("Clusters.csv")

# Define cluster groups and their corresponding labels
cluster_groups = {
    "Group 1": ['s', 'u', 'l'],
    "Group 2": ['p', 'r', 'f', 'm'],
    "Group 3": ['n', 'o', 'q', 't']
}
group_labels = ["FIT", "EXCESS", "DEFICIENCY"]

# Calculate average S and T values and counts for each cluster
avg_s = df_cl.groupby("Cluster")["S"].mean()
avg_t = df_cl.groupby("Cluster")["T"].mean()
cluster_counts = df_cl["Cluster"].value_counts()

# Melt the dataframe to long format for grouped boxplot
df_long = df_cl.melt(id_vars=["Cluster"], value_vars=["P", "J"], var_name="Variable", value_name="Value")

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot grouped box-and-whisker plot for P and J
sns.boxplot(data=df_long, x="Cluster", y="Value", hue="Variable", ax=ax, dodge=True, palette=["lightblue", "orange"])

# Set axis labels and legend text sizes
ax.set_ylabel("P and J Values", fontsize=12)
ax.set_xlabel("Clusters", labelpad=40, fontsize=12)  # Add padding to avoid overlap
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.legend(loc="upper left", fontsize=12)

# Add vertical lines to separate groups and track boundaries for group labels
group_boundaries = [0]  # Track cluster index boundaries for vertical lines
for group_clusters in cluster_groups.values():
    group_boundaries.append(group_boundaries[-1] + len(group_clusters))

for boundary in group_boundaries[1:-1]:  # Skip the first and last boundary
    ax.axvline(x=boundary - 0.5, color="black", linestyle="--", linewidth=1)

# Add group labels at the top of the plot
for i, label in enumerate(group_labels):
    # Calculate the center of each group
    group_start = group_boundaries[i]
    group_end = group_boundaries[i + 1] - 1
    group_center = (group_start + group_end) / 2

    # Add text at the top
    ax.text(group_center, ax.get_ylim()[1] * 1.085, label, 
            ha="center", va="bottom", fontsize=12)

# Add average S and T values and counts under cluster names
for idx, cluster in enumerate(df_cl["Cluster"].unique()):
    avg_tvalue = avg_t.loc[cluster]
    avg_svalue = avg_s.loc[cluster]
    count = cluster_counts.loc[cluster]
    
    # Write average T, S values and data point count
    ax.text(idx, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
            f"T={avg_tvalue:.2f}\nS={avg_svalue:.2f}\nn={count}", 
            ha="center", va="top", fontsize=12)

# Add average S values for each group above the cluster labels
for i, group_label in enumerate(group_labels):
    # Get the clusters in the current group
    group_clusters = cluster_groups[f"Group {i + 1}"]
    
    # Calculate the average S value for the current group
    avg_group_s = avg_s.loc[group_clusters].mean()
    
    # Find the center of the group for placement
    group_start = group_boundaries[i]
    group_end = group_boundaries[i + 1] - 1
    group_center = (group_start + group_end) / 2

    # Add text for the average S value of the group above the cluster labels
    ax.text(group_center, ax.get_ylim()[1] * 1.025, 
            f"S={avg_group_s:.2f}", ha="center", va="bottom", fontsize=12)

# Improve layout
plt.tight_layout()
plt.show()

#Please enter the legend of the title, axis names, axis bounds, increment level on legend, increment on the counters, whether having jittered points, and the colorbar to be appeared in the PRA plot
legend_title='Job Satisfaction'
x_axis='Person'
y_axis='Job'
lb_out=3.5
ub_out=5
inc_levels=0.25
inc_counters=0.1
scat=True
jitter=True
colormap='bwr'

#Plotting the Polynomial Regression Model for the Appendix
plotter(Rmodel, data, var1, var2, labels, lbx, ubx, lby, uby, lb_out, ub_out, inc_levels, inc_counters, legend_title, colormap, x_axis, y_axis, scat, jitter, None)
