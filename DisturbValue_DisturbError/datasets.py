from sklearn.datasets import load_boston
import sklearn
import pandas as pd

def dataloader(dataset):
    if dataset == 'bike':
        df = pd.read_csv('./data/day.csv', index_col='instant')
        Y = pd.DataFrame(df.cnt).to_numpy().reshape((-1, 1))
        X = df.drop(columns=['cnt', 'dteday'])
        X = X.to_numpy()

    elif dataset == 'air':
        df = pd.read_csv('./data/AirQualityUCI.csv', sep=',')
        df = df.dropna(axis=0)
        df['time'] = df.Time.str[:2].astype(int)
        df['month'] = df.Date.str[3:5].astype(int)
        df = df.drop(['Date', 'Time'], axis=1)
  
        cols = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
        Y = pd.DataFrame(df['PT08.S1(CO)']).to_numpy().reshape((-1, 1))
        X = df.drop(columns=cols)
        X = X.to_numpy()

    elif dataset == 'boston':
        bos = load_boston()
        df = pd.DataFrame(bos.data)
        df.columns = bos.feature_names
        df['Price'] = bos.target

        X = df.drop('Price', axis=1)
        Y = df['Price'].to_numpy().reshape((-1, 1))
        X = X.to_numpy()
    
    elif dataset == 'housing':
        df = pd.read_csv('./data/housingprice.csv')
        Y = df.SalePrice.to_numpy().reshape((-1, 1))
        X = df.drop('SalePrice', axis=1)
        X = X.to_numpy()
        
    elif dataset == 'make_sklearn':
        X, Y = sklearn.datasets.make_regression(n_samples=5000, n_features=30, bias=100, noise=100, random_state=0)
        Y = Y.reshape((-1, 1))

    elif dataset == 'superconduct':
        df = pd.read_csv('./data/train.csv', sep=',')
        Y = pd.DataFrame(df.critical_temp).to_numpy().reshape((-1, 1))
        X = df.drop(columns=['critical_temp'])
        X = X.to_numpy()

    elif dataset == 'indoorloc':
        df = pd.read_csv('./data/trainingData.csv', sep=',')
        df = df.dropna()
        Y = pd.DataFrame(df.RELATIVEPOSITION).to_numpy().reshape((-1, 1))
        lst_add=["WAP003","WAP004", "WAP092","WAP093","WAP094","WAP095", "WAP152", "WAP158","WAP159","WAP160","WAP215",
      "WAP217","WAP226","WAP227","WAP238","WAP239","WAP240","WAP241","WAP242","WAP243","WAP244","WAP245","WAP246",
      "WAP247","WAP293", "WAP296", "WAP301", "WAP303","WAP304","WAP307","WAP333","WAP349","WAP353","WAP360",
      "WAP365","WAP416","WAP419","WAP423","WAP429","WAP433", "WAP438","WAP441","WAP442","WAP444","WAP445","WAP451",
      "WAP458","WAP482","WAP485","WAP487","WAP488","WAP491","WAP497","WAP520","WAP254"]
        
        X = df.drop(["SPACEID", "USERID", "RELATIVEPOSITION", "PHONEID", "TIMESTAMP"], axis=1)
        X = X.drop(lst_add, axis=1) 
        X = X.to_numpy()
        #X = np.nan_to_num(X) #added
        #Y = np.nan_to_num(Y) #added

    elif dataset == 'students':
        df_mat = pd.read_csv("./data/student-mat.csv", sep=";")
        df_por = pd.read_csv("./data/student-por.csv", sep=";")
        df=pd.concat([df_mat,df_por], axis=0)
 
        Y = pd.DataFrame(df.G3).to_numpy().reshape((-1, 1))
        X = df.drop(columns=['G3', "school", "sex", "address", "famsize","Pstatus", "Mjob","Fjob","reason","guardian", "schoolsup","famsup","paid","activities","nursery", "higher","internet","romantic","G1","G2"])
        X = X.to_numpy()

    elif dataset == 'energy':
        df = pd.read_csv('./data/energydata_complete.csv', sep=',')
        Y = pd.DataFrame(df.Appliances).to_numpy().reshape((-1, 1))
        X = df.drop(["Appliances", "date"], axis=1)
        X = X.to_numpy()

    elif dataset == 'crime':
        attrib = pd.read_csv('./data/attributes.csv', delim_whitespace = True)
        df = pd.read_csv('communities.data', names = attrib['attributes'])
        
        Y = pd.DataFrame(df.ViolentCrimesPerPop).to_numpy().reshape((-1, 1))
        X = df.drop(columns=["ViolentCrimesPerPop",'state','county','community','communityname','fold', 'OtherPerCap', 'LemasSwornFT', 'LemasSwFTPerPop',
                             'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop',
                             'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits',
                             'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr', 'LemasGangUnitDeploy',
                             'PolicBudgPerPop'], axis=1)
        X = X.to_numpy()

    return X, Y
