from features import load_data, split_data

if __name__ == "__main__":
    p_ticker = "MSFT"
    df = load_data(ticker=p_ticker)
    train_df, test_df, val_df = split_data(df)

    train_df.to_csv("data/"+p_ticker+"_train.csv", index=False)
    test_df.to_csv("data/"+p_ticker+"_test.csv", index=False)
    val_df.to_csv("data/"+p_ticker+"_val.csv", index=False)

    for file in ["data/"+p_ticker+"_train.csv", "data/"+p_ticker+"_test.csv", "data/"+p_ticker+"_val.csv"]:
        with open(file, "r") as f:
            lines = f.readlines()
        with open(file, "w") as f:
            f.writelines([lines[0]] + lines[2:])

    print("Train shape:", train_df.shape)