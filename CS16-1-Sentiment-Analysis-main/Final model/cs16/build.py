

def model_evaluate(model, X_test): 
    import numpy as np
    # predict class with test set
    y_pred_test =  numpy.argmax(model.predict(X_test), axis=1)
    print('Accuracy:\t{:0.1f}%'.format(accuracy_score(np.argmax(y_test,axis=1),y_pred_test)*100))
    
    #classification report
    print('\n')
    print(classification_report(np.argmax(y_test,axis=1), y_pred_test))

    #confusion matrix
    confmat = confusion_matrix(np.argmax(y_test,axis=1), y_pred_test)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()



def late_fusion(image_input, text_input, X_val_image, X_test_image, X_val_text, X_test_text, y_val_polar, y_test_polar):
    from keras.models import Model
    from keras.layers import Input, Dense, concatenate
    
    # image feature extractor
    image_feature_extractor = Model(inputs=image_input, outputs=i)

    # validation data
    val_image_features = image_feature_extractor.predict(X_val_image)
    # test data
    test_image_features = image_feature_extractor.predict(X_test_image)

    # text feature extractor
    text_feature_extractor = Model(inputs=text_input, outputs=t)

    # validation data
    val_text_features = text_feature_extractor.predict(X_val_text)
    # test data
    test_text_features = text_feature_extractor.predict(X_test_text)
    #print(f"val img:{val_image_features.shape} val_text:{val_text_features.shape}")
    # Concatenate image and text features for validation data
    val_features = np.concatenate([val_image_features, val_text_features], axis=1)
    
    # Concatenate image and text features for test data
    test_features = np.concatenate([test_image_features, test_text_features], axis=1)

    # Train logistic regression on concatenated features
    lr = LogisticRegression(multi_class='multinomial', solver='newton-cg')

    # Convert validation labels to one-hot encoded vectors
    y_val_labels = y_val_polar #np.argmax(y_val_image, axis=1)
    y_test_labels = y_test_polar #np.argmax(y_test_image, axis=1)
    lr.fit(val_features, y_val_labels)
    lr_score = lr.score(test_features, y_test_labels)
    print("Late Fusion Test Accuracy: {:.2f}%".format(lr_score * 100))
    from sklearn.metrics import classification_report

    # Predict labels for test data
    y_pred = lr.predict(test_features)

    # Print classification report
    print(classification_report(y_test_labels, y_pred, digits=4))