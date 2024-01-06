import pandas as pd
import matplotlib.pyplot as plt


def concat(df_1, df_2):
    return pd.concat([df_1, df_2])


def read_csv(file):
    return pd.read_csv(file)


if __name__ == '__main__':
    data = read_csv("asset/benchmark_2_1_2024_bias0.csv")

    # Pre-process data

    # 1. Sort by maze number
    data["maze"] = data["maze"].apply(lambda x: int(x.split("_")[1]))
    data = data.sort_values("maze")

    # 2. Remove unwanted models
    random_forest = data[data["model"].str.contains("Random Forest")]
    data = data[~data["model"].str.contains("RandomForestModel")]
    data = data[~data["model"].str.contains("Random Forest")]

    # Save data in different data frames

    # 1. QLearner (is)
    q_learner = data[(data["model"] == "QLearner") | (data["model"] == "Qlearner")]

    # 2. ProgressiveQLearner (contains)
    progressive_learner = data[data["model"].str.contains("ProgressiveQLearner")]

    # 3. Cumulative vs. Single trained
    cumulative_learner = progressive_learner[progressive_learner["model"].str.contains("Cumulative")]
    single_learner = progressive_learner[~progressive_learner["model"].str.contains("Cumulative")]

    # 4. Weka vs New Models (contains)
    weka = progressive_learner[progressive_learner["model"].str.contains("Weka")]
    not_weka = progressive_learner[~progressive_learner["model"].str.contains("Weka")]

    # Plot data (always compare to QLearner, plotting concat)

    # 0. Example plot (bar plot with width = 0.5)
    def plot_comparison(df_1, df_2, column, title, save=None):
        df = pd.concat([df_1, df_2, q_learner])

        # Plot
        df.groupby(["maze", "model"])[
            column
        ].mean().unstack().plot.bar(
            figsize=(10, 5),
            width=0.5,
            title=title,
            xlabel="maze",
            ylabel=column,
        )
        if save is not None:
            plt.savefig(save, dpi=300)
        plt.show()

    # 1. Compare episodes between QLearner and ProgressiveQLearner with Weka non-cumulative
    plot_comparison(
        q_learner,
        # Weka but without cumulative
        weka[~weka["model"].str.contains("Cumulative")],
        "episodes",
        title="Episode comparison between QLearner and ProgressiveQLearner with Weka non-cumulative",
        save="png/episode_comparison_ql_vs_pql_weka_non_cumulative.png",
    )

    # 2. Compare episodes between QLearner and ProgressiveQLearner with Weka cumulative
    plot_comparison(
        q_learner,
        # Weka but with cumulative
        weka[weka["model"].str.contains("Cumulative")],
        "episodes",
        title="Episode comparison between QLearner and ProgressiveQLearner with Weka cumulative",
        save="png/episode_comparison_ql_vs_pql_weka_cumulative.png",
    )

    # 3. Compare episodes between QLearner and ProgressiveQLearner with new models non-cumulative
    # plot_comparison(
    #     q_learner,
    #     # New models but without cumulative
    #     not_weka[~not_weka["model"].str.contains("Cumulative")],
    #     "episodes",
    #     title="Episode comparison between QLearner and ProgressiveQLearner with new models non-cumulative"
    # )

    # 4. Compare episodes between QLearner and ProgressiveQLearner with new models cumulative
    # plot_comparison(
    #     q_learner,
    #     # New models but with cumulative
    #     not_weka[not_weka["model"].str.contains("Cumulative")],
    #     "episodes",
    #     title="Episode comparison between QLearner and ProgressiveQLearner with new models cumulative"
    # )

    # 5. Compare ProgressiveQLearner with Weka and non-Weka non-cumulative
    # plot_comparison(
    #     # Weka
    #     weka[~weka["model"].str.contains("Cumulative")],
    #     # New models
    #     not_weka[~not_weka["model"].str.contains("Cumulative")],
    #     "episodes",
    #     title="Episode comparison between ProgressiveQLearner with Weka and non-Weka"
    # )

    # 6. Compare ProgressiveQLearner with Weka and non-Weka cumulative
    # plot_comparison(
    #     # Weka
    #     weka[weka["model"].str.contains("Cumulative")],
    #     # New models
    #     not_weka[not_weka["model"].str.contains("Cumulative")],
    #     "episodes",
    #     title="Episode comparison between ProgressiveQLearner with Weka and non-Weka cumulative"
    # )

    # plot_comparison(
    #     random_forest,
    #     q_learner,
    #     "episodes",
    #     title="Episode comparison between QLearner and RandomForestModel"
    # )

