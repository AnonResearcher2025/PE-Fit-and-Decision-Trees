#Installing and importing the required libraries and modules
!pip install pandas numpy scikit-learn matplotlib seaborn
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

#Unique label for each variable in the data
labels=['P','J','T','S']

#Lower and upper bound for P (lbx and ubx) and E (lby and uby) values
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

#Setting the maxDepth and minSamples hyperparameters for the DT models
maxDepth=4
minSamples=5

#Building decision tree and polynomial regression models
def build_decision_tree_models(X, y, depth, min_leaf):
    r2=[]
    clf = DecisionTreeRegressor(min_samples_leaf=min_leaf, max_depth=depth, random_state=rn)
    clf.fit(X, y)
    prediction=clf.predict(X)
    model=clf
    r2=r2_score(y, prediction)
    return model, prediction, r2

DTmodel, DTpred, DTr2 =build_decision_tree_models(data.iloc[:,0:2], data.iloc[:,-1], depth=maxDepth, min_leaf=minSamples)
DTmodel_cont, DTpred_cont, DTr2_cont =build_decision_tree_models(data.iloc[:,0:-1], data.iloc[:,-1], depth=maxDepth, min_leaf=minSamples)

def build_polynomial_regression_model(X, y):
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    prediction=regr.predict(X)
    r2=r2_score(prediction, y)
    return regr, prediction, r2

Rmodel, Rpred, Rr2 =build_polynomial_regression_model(regr_data.iloc[:,0:5], regr_data.iloc[:,-1])

#Plotter function to produce PRA and contour plots
def plotter(model, data, var1, var2, labels, lbx, ubx, lby, uby, lb_out, ub_out, inc_levels, inc_counters, legend_title, colormap, x_axis, y_axis, scat, jitter, ax, r2):
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
        data_jit[labels[0]+'_jittered']=data.iloc[:, 0]+np.random.randn(len(data.iloc[:, 0]))*0.002*(ubx-lbx)
        data_jit.loc[data_jit[labels[0]+'_jittered']>=ubx, labels[0]+'_jittered']=ubx-abs(np.random.randn()*0.002*(ubx-lbx))
        data_jit.loc[data_jit[labels[0]+'_jittered']<=lbx, labels[0]+'_jittered']=lbx+abs(np.random.randn()*0.002*(ubx-lbx))
        data_jit[labels[1]+'_jittered']=data.iloc[:, 1]+np.random.randn(len(data.iloc[:, 1]))*0.002*(uby-lby)
        data_jit.loc[data_jit[labels[1]+'_jittered']>=uby, labels[1]+'_jittered']=uby-abs(np.random.randn()*0.002*(uby-lby))
        data_jit.loc[data_jit[labels[1]+'_jittered']<=lby, labels[1]+'_jittered']=lby+abs(np.random.randn()*0.002*(uby-lby))

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
        if r2 is not None:
            plt.text(0, -2.7, f'R²={r2:.2f}', ha='center', va='center', fontsize=10)
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
        if r2 is not None:
            ax.text(0.5, -0.35, f'R²={r2:.2f}', ha='center', va='center', fontsize=10, transform=ax.transAxes)
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
fig=plt.figure(figsize=(80,40))
_=plot_tree(DTmodel, feature_names=labels[0:2], filled=True)
plt.show()
plt.close(fig)

#Setting the legend of the title, axis names, colorbar bounds, increment level for colorbar, colormap for the colorbar, whether having scatter points and jittered points for Figure 2
legend_title='Job Satisfaction'
x_axis='Person'
y_axis='Job'
lb_out=3.5
ub_out=5
inc=0.25
colormap='gray'
scat=True
jitter=True

#Plotting Figure 2
plotter(DTmodel, data, var1, var2, labels, lbx, ubx, lby, uby, lb_out, ub_out, inc, inc, legend_title, colormap, x_axis, y_axis, scat, jitter, None, DTr2)

#Plotting Figure 6
fig=plt.figure(figsize=(80,40))
_=plot_tree(DTmodel_cont, feature_names=labels[0:-1], filled=True)
plt.show()
plt.close(fig)

#Setting the critical values of the contextual variable and its name
crit_val=[2.5,9.5]
Cvar='Tenure'

#Preparing the grouped data
crit_val.sort()
crit_val.append(np.inf)
groups=[]
lb=-np.inf

for i in range(0,len(crit_val)):
    ub=crit_val[i]
    globals()[f"dataG{i+1}"]=data[(data[Cvar]>lb)&(data[Cvar]<=ub)]
    lb=ub

#Building the DT models for the grouped data
for i in range(len(crit_val)):
    globals()[f"DTmodelG{i+1}"], globals()[f"DTpredG{i+1}"], globals()[f"DTr2G{i+1}"]=build_decision_tree_models(globals()[f"dataG{i+1}"].iloc[:, 0:2], 
                                                                                       globals()[f"dataG{i+1}"].iloc[:, -1], depth=maxDepth, min_leaf=minSamples)

#Plotting the Models for Different Groups
figure, axis=plt.subplots(1,3,sharex=True,sharey=True,subplot_kw=dict(aspect='equal'))

plotter(DTmodelG1, dataG1, var1, var2, labels, lbx, ubx, lby, uby, lb_out, ub_out, inc, inc, legend_title, colormap, x_axis, y_axis, scat, jitter, axis[0], DTr2G1)
plotter(DTmodelG2, dataG2, var1, var2, labels, lbx, ubx, lby, uby, lb_out, ub_out, inc, inc, legend_title, colormap, x_axis, y_axis, scat, jitter, axis[1], DTr2G2)
im=plotter(DTmodelG3, dataG3, var1, var2, labels, lbx, ubx, lby, uby, lb_out, ub_out, inc, inc, legend_title, colormap, x_axis, y_axis, scat, jitter, axis[2], DTr2G3)

cbar2=plt.colorbar(im, ax=axis[:], location='top', shrink=0.4, ticks=np.arange(lb_out, ub_out+0.00001, inc))
cbar2.ax.tick_params(labelsize=8)
cbar2.set_label("Job Satisfaction", rotation=0, loc='center', size=10)
for i, label in enumerate(["0 ≤ T ≤ 2.5", "2.5 < T ≤ 9.5", "9.5 < T ≤ 38"]): 
    axis[i].text(0.5, -0.5, label, ha='center', va='center', fontsize=10, 
                 transform=axis[i].transAxes)

#Importing the clusters data
df_cl=pd.read_csv("Clusters.csv")

#Define cluster groups and their corresponding labels
cluster_groups={
    "Group 1":['s', 'u', 'l'],
    "Group 2":['p', 'r', 'f', 'm'],
    "Group 3":['n', 'o', 'q', 't']}
group_labels = ["FIT", "EXCESS", "DEFICIENCY"]

#Calculate average S and T values and sample sizes for each cluster
avg_s=df_cl.groupby("Cluster")["S"].mean()
avg_t=df_cl.groupby("Cluster")["T"].mean()
cluster_counts=df_cl["Cluster"].value_counts()

#Melt the dataframe to long format for grouped boxplot
df_long=df_cl.melt(id_vars=["Cluster"], value_vars=["P", "J"], var_name="Variable", value_name="Value")

#Set up the figure and axis
fig, ax=plt.subplots(figsize=(12, 6))

#Plot grouped box-and-whisker plot for P and J
sns.boxplot(data=df_long, x="Cluster", y="Value", hue="Variable", ax=ax, dodge=True, palette=["lightgrey", "dimgrey"])

#Set axis labels and legend text sizes
ax.set_ylabel("P and J Values", fontsize=12)
ax.set_xlabel("Clusters", labelpad=40, fontsize=12)
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.legend(loc="upper left", fontsize=12)

#Add vertical lines to separate groups and track boundaries for group labels
group_boundaries=[0]
for group_clusters in cluster_groups.values():
    group_boundaries.append(group_boundaries[-1]+len(group_clusters))

for boundary in group_boundaries[1:-1]:  # Skip the first and last boundary
    ax.axvline(x=boundary-0.5, color="black", linestyle="--", linewidth=1)

#Add group labels at the top of the plot
for i, label in enumerate(group_labels):
    group_start=group_boundaries[i]
    group_end=group_boundaries[i+1]-1
    group_center=(group_start+group_end)/2

    ax.text(group_center, ax.get_ylim()[1]*1.085, label, 
            ha="center", va="bottom", fontsize=12)

#Add average S and T values and counts under cluster names
for idx, cluster in enumerate(df_cl["Cluster"].unique()):
    avg_tvalue=avg_t.loc[cluster]
    avg_svalue=avg_s.loc[cluster]
    count=cluster_counts.loc[cluster]
    ax.text(idx, ax.get_ylim()[0]-(ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, 
            f"T={avg_tvalue:.2f}\nS={avg_svalue:.2f}\nn={count}", 
            ha="center", va="top", fontsize=12)

#Add average S values for each group above the cluster labels
for i, group_label in enumerate(group_labels):
    group_clusters=cluster_groups[f"Group {i + 1}"]
    avg_group_s=avg_s.loc[group_clusters].mean()
    group_start=group_boundaries[i]
    group_end=group_boundaries[i + 1]-1
    group_center=(group_start + group_end)/2
    ax.text(group_center, ax.get_ylim()[1]*1.025, 
            f"S={avg_group_s:.2f}", ha="center", va="bottom", fontsize=12)

#Improve layout
plt.tight_layout()
plt.show()