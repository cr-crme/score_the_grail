from score_the_grail import KinematicData, NormativeData


def main():
    # TODO - Extract the steps from the non-normalized data?
    kd_exported = KinematicData.from_csv("data/pilot/1113-Gait analysis - Balloon animals.csv")
    kd_all = KinematicData.from_mox("data/pilot/1113-Gait analysis - Balloon animals.mox")

    norm = KinematicData.from_normative_data(file=NormativeData.CROUCHGAIT)

    from matplotlib import pyplot as plt

    plt.plot(norm.data["Rotation LKneeFlex"].to_numpy())
    plt.show()

    kd_exported_normalized = KinematicData.from_normalized_csv(
        "data/pilot/1113-Gait analysis - Balloon animals_normalized.csv"
    )
    gps = kd_exported_normalized.gps(normative_data=NormativeData.CROUCHGAIT)
    print(gps.data)


if __name__ == "__main__":
    main()
