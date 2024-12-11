from matplotlib import pyplot as plt
from score_the_grail import KinematicData, NormativeData


def main():
    # kd_exported = KinematicData.from_csv("data/pilot/1113-Gait analysis - Balloon animals.csv")
    # kd_all = KinematicData.from_mox("data/pilot/1113-Gait analysis - Balloon animals.mox")
    kd_exported_normalized = KinematicData.from_normalized_csv("data/pilot/CS_2036100_POST.csv")
    kd_normative = KinematicData.from_normative_data(NormativeData.NORMAL)

    gps = kd_exported_normalized.gps(normative_data=NormativeData.NORMAL)
    print(f"GPS for left side: {gps.left.to_numpy:.3f} and for right side: {gps.right.to_numpy:.3f}")

    plt.figure("Kinematic Data")
    dofs_to_plot = ["HipFlex", "KneeFlex", "AnkleFlex"]
    sides = ["left", "right"]
    for j, dof in enumerate(dofs_to_plot):
        plt.subplot(len(dofs_to_plot), 1, j + 1)
        plt.title(dof)
        plt.axis("off")
        y_lims = []
        for i, side in enumerate(sides):
            plt.subplot(len(dofs_to_plot), 2, 2 * j + i + 1)
            plt.title(side)
            plt.plot(kd_exported_normalized.get_side(side)[dof].to_numpy, linestyle="--", color="b")
            plt.plot(kd_normative.get_side(side)[dof].to_numpy, linestyle="-", color="k")
            y_lims.append(plt.ylim())
        new_y_lim = (min([y[0] for y in y_lims]), max([y[1] for y in y_lims]))
        for i in range(len(sides)):
            plt.subplot(len(dofs_to_plot), 2, 2 * j + i + 1)
            plt.ylim(new_y_lim)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
