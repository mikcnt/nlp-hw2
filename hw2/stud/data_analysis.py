from stud.dataset import read_data


def count_polarities(dataset, mode):
    assert mode in ["targets", "categories"]

    if mode == "targets":
        targets = [x["targets"] for x in dataset]

        frequencies = {"positive": 0, "negative": 0, "neutral": 0, "conflict": 0}

        for target in targets:
            for t in target:
                frequencies[t[2]] += 1

        return frequencies

    if mode == "categories":
        categories = [x["categories"] for x in dataset]

        category_frequencies = {
            "anecdotes/miscellaneous": 0,
            "price": 0,
            "food": 0,
            "ambience": 0,
            "service": 0,
        }
        polarity_frequencies = {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "conflict": 0,
        }

        for category in categories:
            for t in category:
                category_frequencies[t[0]] += 1
                polarity_frequencies[t[1]] += 1

        return category_frequencies, polarity_frequencies


if __name__ == "__main__":
    # dataset paths
    restaurants_train_path = "../../data/restaurants_train.json"
    restaurants_dev_path = "../../data/restaurants_dev.json"
    laptops_train_path = "../../data/laptops_train.json"
    laptops_dev_path = "../../data/laptops_dev.json"
    # read data
    restaurants_train_data = read_data(restaurants_train_path)
    restaurants_dev_data = read_data(restaurants_dev_path)
    laptops_train_data = read_data(laptops_train_path)
    laptops_dev_data = read_data(laptops_dev_path)
    train_data = restaurants_train_data + laptops_train_data
    dev_data = restaurants_dev_data + laptops_dev_data

    # count frequencies of polarities for targets (A+B)
    laptops_train_frequencies = count_polarities(laptops_train_data, mode="targets")
    laptops_dev_frequencies = count_polarities(laptops_dev_data, mode="targets")
    restaurants_train_frequencies = count_polarities(
        restaurants_train_data, mode="targets"
    )
    restaurants_dev_frequencies = count_polarities(restaurants_dev_data, mode="targets")

    all_train_freqs = {
        k: laptops_train_frequencies[k] + restaurants_train_frequencies[k]
        for k in laptops_train_frequencies.keys()
    }
    all_dev_freqs = {
        k: laptops_dev_frequencies[k] + restaurants_dev_frequencies[k]
        for k in laptops_dev_frequencies.keys()
    }

    # count frequencies of polarities for categories (C+D, only restaurants)
    (
        restaurants_train_freqs_categories,
        restaurants_train_freqs_polarities,
    ) = count_polarities(restaurants_train_data, mode="categories")
    (
        restaurants_dev_freqs_categories,
        restaurants_dev_freqs_polarities,
    ) = count_polarities(restaurants_dev_data, mode="categories")

    # print all
    print("#" * 10, "TARGETS", "#" * 10)
    print("LAPTOPS TRAIN:", laptops_train_frequencies)
    print("LAPTOPS DEV:", laptops_dev_frequencies)
    print("RESTAURANTS TRAIN:", restaurants_train_frequencies)
    print("RESTAURANTS DEV:", restaurants_dev_frequencies)
    print("ALL TRAIN:", all_train_freqs)
    print("ALL DEV:", all_dev_freqs)

    print("\n\n")

    print("#" * 10, "CATEGORIES", "#" * 10)
    print("RESTAURANTS TRAIN CATEGORIES:", restaurants_train_freqs_categories)
    print("RESTAURANTS TRAIN POLARITIES:", restaurants_train_freqs_polarities)
    print("RESTAURANTS DEV CATEGORIES:", restaurants_dev_freqs_categories)
    print("RESTAURANTS DEV POLARITIES:", restaurants_dev_freqs_polarities)
