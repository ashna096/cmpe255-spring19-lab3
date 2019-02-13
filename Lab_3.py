from sklearn.metrics import accuracy_score


def load_data():
    data = []
    data_labels = []
    with open(
            "C:/Users/ASHNA GUPTA/Desktop/New folder/cmpe255-spring19-master/cmpe255-spring19-master/lab3/pos_tweets.txt",
            encoding="utf-8") as f:
        for i in f:
            data.append(i)
            data_labels.append('pos')

    with open(
            "C:/Users/ASHNA GUPTA/Desktop/New folder/cmpe255-spring19-master/cmpe255-spring19-master/lab3/neg_tweets.txt",
            encoding="utf-8") as f:
        for i in f:
            data.append(i)
            data_labels.append('neg')

    return data, data_labels


def transform_to_features(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        analyzer='word',
        lowercase=False,
    )
    features = vectorizer.fit_transform(data)
    features_nd = features.toarray()
    return features_nd


def train_then_build_model(data_labels, features_nd, data):
    from sklearn.cross_validation import train_test_split
    # TODO : set training % to 80%.
    X_train, X_test, y_train, y_test = train_test_split(
        features_nd,
        data_labels,
        train_size=0.80,
        random_state=1234)

    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression()

    log_model = log_model.fit(X=X_train, y=y_train)
    y_pred = log_model.predict(X_test)

    for i in range(0, 10):
        ind = features_nd.tolist().index(X_test[i].tolist())
        print("::{}::{}".format(y_pred[i], data[ind].strip()))

    # print accuracy
    from sklearn.metrics import accuracy_score
    # TODO
    val = accuracy_score(y_pred, y_test)

    print("Accuracy={}".format(val))


def process():
    data, data_labels = load_data()
    features_nd = transform_to_features(data)
    train_then_build_model(data_labels, features_nd, data)


process()