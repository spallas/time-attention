from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def barchart(data, i, types, label):
    bars = len(types) - 1
    plt.bar(
        range(bars),
        [d[i] for d in data][:-1],
        0.35,
        color=list(mcolors.TABLEAU_COLORS)[: bars],
    )
    plt.axhline(y=data[-1][i], zorder=3, color="red", label='da-rnn')
    plt.xticks(range(bars), types[:-1])
    plt.ylabel(label)
    plt.legend()
    plt.title("Attention comparison (the lower the better)")
    plt.show()


if __name__ == "__main__":
    print()
    with open("attn_result.csv") as f:
        header = next(f).strip().split(",")
        data = []
        for l in f:
            d = l.strip().split(",")
            d[1:] = [float(e) for e in d[1:]]
            data.append(d)
    types = [
        "no-" + Path(e[0]).with_suffix("").name.split("_")[-1] for e in data
    ]
    barchart(data, -3, types, "RMSE")
