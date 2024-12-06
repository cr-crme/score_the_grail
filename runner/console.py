from score_the_grail import KinematicData, NormativeData


def main():
    kd_exported = KinematicData.from_csv("data/pilot/1113-Gait analysis - Balloon animals.csv")
    kd_all = KinematicData.from_mox("data/pilot/1113-Gait analysis - Balloon animals.mox")
    kd_exported_normalized = KinematicData.from_normalized_csv(
        "data/pilot/1113-Gait analysis - Balloon animals_normalized.csv"
    )
    kd_normative = KinematicData.from_normative_data(file=NormativeData.CROUCHGAIT)

    # from matplotlib import pyplot as plt

    # plt.plot(kd_exported_normalized.right["KneeFlex"].right.data)
    # plt.show()

    gps = kd_exported_normalized.gps(normative_data=NormativeData.CROUCHGAIT)
    print(gps.data)


if __name__ == "__main__":
    main()
