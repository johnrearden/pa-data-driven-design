import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from src.data_management import load_airplane_data, load_pkl_file


def page_ml_cluster_body():
    # load cluster analysis files and pipeline
    version = 'v1'
    cluster_pipe = load_pkl_file(
        f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl")
    cluster_silhouette = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_silhouette.png")
    features_to_cluster = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/v1/features_define_cluster.png")
    cluster_cluster_vs_engine_type = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/v1/cluster_vs_engine_type.png")
    features_cluster_vs_multi_engine = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/v1/cluster_vs_multi_engine.png")
    cluster_clusters_profile_engine_type = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/v1/clusters_profile_engine_type.csv")
    cluster_clusters_profile_multi_engine = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/v1/clusters_profile_multi_engine.csv")
    cluster_features = (pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
                        .columns
                        .to_list()
                        )


    # dataframe for cluster_distribution_per_variable()
    df_multi_engine_vs_clusters = load_airplane_data().filter(['Multi_Engine'], axis=1)
    print(df_multi_engine_vs_clusters)
    df_multi_engine_vs_clusters['Clusters'] = cluster_pipe['model'].labels_

    st.write("### ML Pipeline: Cluster Analysis")
    # display pipeline training summary conclusions
    st.info(
        f"* We could have refitted the cluster pipeline using drastically fewer variables, with equivalent result."
        f" However the outcome of the exercise proved disappointing since the only significant"
        f" cluster found was trivial and due to if the airplane had a Performance enhancing modification to"
        f" it's Engine (TP mods) or not.  \n"
        f"* The pipeline average silhouette score is 0.87 which is very good and high above the target of >0.5."
        f" The number of clusters are also less than the criteria of < 5."
    )

    st.write("---")

    st.write("#### Cluster ML Pipeline steps")
    st.write(cluster_pipe)

    st.write("---")

    st.write("#### Clusters Silhouette Plot")

    statement = (
        f"* A **relatively high Average Silhouette Score** - overall high clustering"
        f" quality, well-clustered points with minimal overlap or misclassification.  \n"

        f"* Silhouette Score for both clusters are **well above the average silhouette"
        f" score** - strong internal cohesion and separation from each other.  \n"

        f"* **Uniform silhouette thickness and shape** for both clusters - well-formed clusters"
        f" with consistent cohesion and separation across data points."

        f"**No thin or overly elongated silhouettes** -"
        f" good definition.  \n"

        f"* Silhouette scores for the two clusters are **well separated with"
        f" a gap between them** - distinctly separated"
        f" (far apart) clusters with no overlap  \n"
    )
    st.success(statement)

    st.image(cluster_silhouette)

    cluster_distribution_per_variable(df=df_multi_engine_vs_clusters, target='Multi_Engine')

    st.write("#### Most important features to define a cluster")
    st.image(features_to_cluster)

    st.write("---")

    # text based on "8 - Modeling and Evaluation - Cluster" notebook conclusions
    statement = (
        f"* Based on the profile results we can label the two cluster in the following fashion:\n"
        f"* **Cluster 0** (no TP mods) made up"
        f" by all Engine Types (piston, propjet and jet) and multi/single engine airplanes in a relatively even distribution.\n"
        f"* **Cluster 1** (TP mods) almost more multi engine than single engined airplanes and"
        f" almost exclusively made up by"
        f" Piston powered Airplanes (Airplane Type: piston) indicating that the TP mods is a feature only"
        f" relevant for or used on Piston Powered Engines.\n"
    )
    st.success(statement)

    # hack to not display the index in st.table() or st.write()
    cluster_clusters_profile_multi_engine.index = [" "] * len(cluster_clusters_profile_multi_engine)
    st.table(cluster_clusters_profile_multi_engine)
    cluster_clusters_profile_engine_type.index = [" "] * len(cluster_clusters_profile_engine_type)
    st.table(cluster_clusters_profile_engine_type)

# code coped from "07 - Modeling and Evaluation - Cluster Sklearn" notebook - under "Cluster Analysis" section
def cluster_distribution_per_variable(df, target):

    df_bar_plot = df.value_counts(["Clusters", target]).reset_index()
    df_bar_plot.columns = ['Clusters', target, 'Count']
    df_bar_plot[target] = df_bar_plot[target].astype('object')

    st.write(f"#### Clusters distribution across {target} levels")
    fig = px.bar(df_bar_plot, x='Clusters', y='Count',
                 color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    # we replaced fig.show() for a streamlit command to render the plot
    st.plotly_chart(fig)

    df_relative = (df
                   .groupby(["Clusters", target])
                   .size()
                   .groupby(level=0)
                   .apply(lambda x:  100*x / x.sum())
                   .reset_index()
                   .sort_values(by=['Clusters'])
                   )
    df_relative.columns = ['Clusters', target, 'Relative Percentage (%)']

    st.write(f"#### Relative Percentage (%) of {target} in each cluster")
    fig = px.line(df_relative, x='Clusters', y='Relative Percentage (%)',
                  color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    fig.update_traces(mode='markers+lines')
    # we replaced fig.show() for a streamlit command to render the plot
    st.plotly_chart(fig)
