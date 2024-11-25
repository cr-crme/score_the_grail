from score_the_grail import KinematicData


def main():
    kd_exported = KinematicData.from_csv("data/pilot/1113-Gait analysis - Balloon animals.csv")
    kd_noralized = KinematicData.from_normalized_csv("data/pilot/1113-Gait analysis - Balloon animals_normalized.csv")
    kd_all = KinematicData.from_mox("data/pilot/1113-Gait analysis - Balloon animals.mox")

    # Print the first 5 rows of the data
    print(kd_exported.data.head())
    print(kd_noralized.data.head())
    print(kd_all.data.head())


if __name__ == "__main__":
    main()
