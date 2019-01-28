from pathlib import Path

import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = []

    with open("results.csv") as f:
        header = next(f).strip().split(",")
        for line in f:
            fields = line.strip().split(",")
            conf_name = str(Path(fields[0]).with_suffix(""))
            params = conf_name.split("_")[1:]
            t = int(params[0].strip("T"))
            m_p = int(params[1][params[1].find("p") + 1 :])
            # test_rmse,test_mae,test_mape
            test_scores = [float(field) for field in fields[-3:]]

            data.append((conf_name, t, m_p, *test_scores))

    grouped_by_t = {}
    for e in data:
        grouped_by_t[e[1]] = grouped_by_t.get(e[1], [])
        grouped_by_t[e[1]].append(e)

    plots = []
    for t in sorted(grouped_by_t):
        plots.append(
            [e[-3] for e in sorted(grouped_by_t[t], key=lambda e: e[2])]
        )

    [m16, m32, m64, m128, m256] = plt.plot(
        sorted(list(grouped_by_t.keys())), plots
    )
    plt.legend(
        [m16, m32, m64, m128, m256],
        ["m/p 16", "m/p 32", "m/p 64", "m/p 128", "m/p 256"],
    )
    plt.xticks(sorted(list(grouped_by_t.keys())))
    plt.xlabel("Window length")
    plt.ylabel("RMSE")
    plt.show()
