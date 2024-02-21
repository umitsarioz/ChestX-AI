if [ ! -d models ]
then
     echo "Models files are downloading..." &&
     mkdir models &&
     gdown 1--8Ge66xPQSbmWEjUCI7sKFyjdkpJ6Sm --output models/token_dct.pkl &&
     gdown 1-07A9vgDMMhP2RjWPHt_HtYpNFTM5_PF --output models/xception.keras &&
     gdown 1-409KKZf3jj7sew883DPayWF3hFwysro --output models/basic_model.h5
else
     echo "Directory models is already exists"
fi

