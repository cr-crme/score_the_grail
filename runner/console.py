from matplotlib import pyplot as plt
from score_the_grail import KinematicData, NormativeData


def main():
    # kd_exported = KinematicData.from_csv("data/pilot/1113-Gait analysis - Balloon animals.csv")
    # kd_all = KinematicData.from_mox("data/pilot/1113-Gait analysis - Balloon animals.mox")
    kd_exported_normalized = KinematicData.from_normalized_csv("data/pilot/CS_2036100_POST.csv")
    kd_normative = KinematicData.from_normative_data(NormativeData.NORMAL)

    gps = kd_exported_normalized.gps(normative_data=NormativeData.NORMAL)
    print(f"GPS for left side: {gps.left.to_numpy:.3f} and for right side: {gps.right.to_numpy:.3f}")

    dof_to_plot = "HipFlex"
    plt.figure("Right side")
    plt.plot(kd_exported_normalized.right[dof_to_plot].to_numpy, "b")
    plt.plot(kd_normative.right[dof_to_plot].to_numpy, "k")
    plt.figure("Left side")
    plt.plot(kd_exported_normalized.left[dof_to_plot].to_numpy, "b")
    plt.plot(kd_normative.left[dof_to_plot].to_numpy, "k")
    plt.show()


if __name__ == "__main__":
    main()
