from score_the_grail import KinematicData, NormativeData


def main():
    kd_exported = KinematicData.from_csv("data/pilot/1113-Gait analysis - Balloon animals.csv")
    kd_all = KinematicData.from_mox("data/pilot/1113-Gait analysis - Balloon animals.mox")
    kd_exported_normalized = KinematicData.from_normalized_csv(
        "data/pilot/1113-Gait analysis - Balloon animals_normalized.csv"
    )
    kd_normative = KinematicData.from_normative_data(file=NormativeData.CROUCHGAIT)

    gps = kd_exported_normalized.gps(normative_data=NormativeData.CROUCHGAIT)
    print(f"GPS for left side: {gps.left.to_numpy:.3f} and for right side: {gps.right.to_numpy:.3f}")

    from matplotlib import pyplot as plt

    plt.figure("Right side")
    plt.plot(kd_exported_normalized.right["KneeFlex"].to_numpy)
    plt.figure("Left side")
    plt.plot(kd_exported_normalized.left["KneeFlex"].to_numpy)
    plt.show()


if __name__ == "__main__":
    main()
