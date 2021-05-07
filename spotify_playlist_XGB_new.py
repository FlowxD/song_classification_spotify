import spotipy
import pandas as pd
import seaborn as sns
import spotipy.oauth2 as oauth2
import sklearn

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import plot_confusion_matrix

def generate_token():
    """ Generate the token. Please respect these credentials :) """
    credentials = oauth2.SpotifyClientCredentials(
        client_id='5e22c02862e0444aac0351ee2aca9e2b',
        client_secret='dd2407b5b3984fa38bf707b883e34bea')
    token = credentials.get_access_token()
    return token



import pandas as pd 
import spotipy 
sp = spotipy.Spotify() 
from spotipy.oauth2 import SpotifyClientCredentials 
cid ="5e22c02862e0444aac0351ee2aca9e2b" 
secret = "dd2407b5b3984fa38bf707b883e34bea" 
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
sp.trace=False 
playlist = sp.user_playlist("SPOTIFY_USERNAME", "62y3BHKehWnb1hlaPclDAA") 

songs = playlist["tracks"]["items"] 
tracks = playlist["tracks"]

ids = [] 
while tracks['next']:
    tracks = sp.next(tracks)
    for item in tracks["items"]:
        if (item['track']['id'] is not None):
            ids.append(item['track']['id'])


features = []
for i in range(0,len(ids),50):    
    audio_features = sp.audio_features(ids[i:i+50])
    for track in audio_features:
        features.append(track)  
df_final = pd.read_csv (r'C:/Users/Mandar Joshi/Desktop/spotify_classification/df_final.csv')




df_hip = pd.DataFrame(columns=['danceability','loudness','speechiness','acousticness','liveness','instrumentalness','type'])

df_hip.to_csv(r'C:/Users/Mandar Joshi/Desktop/spotify_classification/df_hip.csv', index=False)
for i in features:
    danceability = i.get("danceability")
    loudness = i.get("loudness")
    speechiness = i.get("speechiness")
    acousticness = i.get("acousticness")
    liveness = i.get("liveness")
    instrumentalness = i.get("instrumentalness")
    type1 = 5
    df_hip.loc[len(df_hip.index)] = [danceability,loudness,speechiness,acousticness,liveness,instrumentalness,type1] 




frames = [df_final,df_jazz,df_hip]
df_final = pd.concat(frames)


df_final.to_csv(r'C:/Users/Mandar Joshi/Desktop/spotify_classification/df_final_5.csv', index=False)




from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# split data into X and y
X = df_final.iloc[:, :-1].values
Y = df_final.iloc[:, -1].values

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


print(multilabel_confusion_matrix(y_test, y_pred, labels=[1.0, 2.0, 3.0, 4.0, 5.0]))

class_names = [1.0, 2.0, 3.0, 4.0, 5.0]
import matplotlib.pyplot as plt

disp = plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues)
plt.show()




import pickle
pickle.dump(model, open(r'C:/Users/Mandar Joshi/Desktop/spotify_classification/spotify_model2.pkl', 'wb'))













