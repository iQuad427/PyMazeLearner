import pandas as pd
import matplotlib.pyplot as plt


def read_csv():
    df = pd.read_csv("benchmark_31_12_2023.csv")
    df_2 = pd.read_csv("out.csv")
    return pd.concat([df, df_2])


if __name__ == '__main__':
    data = read_csv()

    # Sort by maze number
    data["maze"] = data["maze"].apply(lambda x: int(x.split("_")[1]))
    data = data.sort_values("maze")

    # Remove the model containing random forest
    data = data[~data["model"].str.contains("RandomForest")]

    # Recover all lines with QLearner as model
    q_learner = data[data["model"] == "QLearner"]
    progressive_q_learner = data[data["model"].str.contains("ProgressiveQLearner")]

    cumulative_q_learner = data[data["model"].str.contains("Cumulative")]
    single_trained_learner = progressive_q_learner[~progressive_q_learner["model"].str.contains("Cumulative")]

    # Bar plot of episodes per maze and per model for QLearner and SingleTrainedLearner
    # Merge the two dataframes q_learner and single_trained_learner using concat

    single_vs_q_learner = pd.concat([q_learner, single_trained_learner])

    single_vs_q_learner.groupby(["maze", "model"])["steps"].mean().unstack().plot.bar()
    plt.savefig("single_vs_q_learner.png")
    plt.show()

    cumulative_vs_q_learner = pd.concat([q_learner, cumulative_q_learner])

    cumulative_vs_q_learner.groupby(["maze", "model"])["steps"].mean().unstack().plot.bar()
    plt.savefig("cumulative_vs_q_learner.png")
    plt.show()

    # # Bar plot of episodes per maze and per model
    # data.groupby(["maze", "model"])["episodes"].mean().unstack().plot.bar()
    # plt.show()
    #
    # # Bar plot of steps per maze and per model
    # data.groupby(["maze", "model"])["steps"].mean().unstack().plot.bar()
    # plt.show()
