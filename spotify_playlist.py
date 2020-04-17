import spotipy
import pandas as pd
import seaborn as sns
import spotipy.oauth2 as oauth2

def generate_token():
    """ Generate the token. Please respect these credentials :) """
    credentials = oauth2.SpotifyClientCredentials(
        client_id='5e22c02862e0444aac0351ee2aca9e2b',
        client_secret='dd2407b5b3984fa38bf707b883e34bea')
    token = credentials.get_access_token()
    return token




def write_tracks(text_file, tracks):
    ids=[]
#    with open(text_file, 'a') as file_out:
    while True:    
        for item in tracks['items']:
            if 'track' in item:
                track = item['track']
            else:
                track = item
            try:
                track_url = track['external_urls']['spotify']
                ids.append(track_url[-22:len(track_url)])    
#                file_out.write(track_url + '\n')
                
                
            except KeyError:
                print(u'Skipping track {0} by {1} (local only?)'.format(
                        track['name'], track['artists'][0]['name']))
            # 1 page = 50 results
            # check if there are more pages
        if tracks['next']:
            tracks = spotify.next(tracks)
        else:
            break
    id=ids[0]
    print(id)
    print(type(id))
    columns = ['danceability','loudness','speechiness','acousticness','liveness','instrumentalness']
    df_s = pd.DataFrame(columns=columns)
    for id in ids:
        
        client_credentials_manager = oauth2.SpotifyClientCredentials(client_id='5e22c02862e0444aac0351ee2aca9e2b', client_secret='dd2407b5b3984fa38bf707b883e34bea')
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        features = sp.audio_features(id)
        print(features)
        lst = list(get_all_values(features))

        df2 = {'danceability': lst[0], 'loudness': lst[3], 'speechiness': lst[5], 'acousticness': lst[6], 'liveness': lst[8], 'instrumentalness': lst[7]}
        df_s = df_s.append(df2, ignore_index=True)
    print(df_s)   
    return df_s
       
        



def get_all_values(d):
    if isinstance(d, dict):
        for v in d.values():
            yield from get_all_values(v)
    elif isinstance(d, list):
        for v in d:
            yield from get_all_values(v)
    else:
        yield d 

#rap playlist       
def write_playlist(username, playlist_id):
    results = spotify.user_playlist(username, playlist_id,
                                    fields='tracks,next,name')
    text_file = u'{0}.txt'.format(results['name'], ok='-_()[]{}')
    print(u'Writing {0} tracks to {1}'.format(
            results['tracks']['total'], text_file))
    tracks = results['tracks']
    df_rap = write_tracks(text_file, tracks)
    return df_rap


token = generate_token()
spotify = spotipy.Spotify(auth=token)
    
#sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# example playlist
df_rap = write_playlist('abc', '1aHaA2sJVYDCV0UZR8YgVJ')

df_rap['y'] = 'rap'
#---------------------------------------------

#lofi playlist
def write_playlist(username, playlist_id):
    results = spotify.user_playlist(username, playlist_id,
                                    fields='tracks,next,name')
    text_file = u'{0}.txt'.format(results['name'], ok='-_()[]{}')
    print(u'Writing {0} tracks to {1}'.format(
            results['tracks']['total'], text_file))
    tracks = results['tracks']
    df_lofi = write_tracks(text_file, tracks)
    return df_lofi

df_lofi = write_playlist('abc', '43udSsOeQC1mlUYf18fb2J')
df_lofi['y'] = 'lofi'
#---------------------------------------------


#dance playlist
def write_playlist(username, playlist_id):
    results = spotify.user_playlist(username, playlist_id,
                                    fields='tracks,next,name')
    text_file = u'{0}.txt'.format(results['name'], ok='-_()[]{}')
    print(u'Writing {0} tracks to {1}'.format(
            results['tracks']['total'], text_file))
    tracks = results['tracks']
    df_dance = write_tracks(text_file, tracks)
    return df_dance

df_dance = write_playlist('abc', '1hFHtd8MjxfjGMG2dfR2GG')
df_dance['y'] = 'dance'
#---------------------------------------------
#run above code for each playlist of different genres 

#run below code to combine each data frame of genres 
df_rap['y'] = 1
df_lofi['y'] = 2
df_dance['y'] = 3

frames = [df_lofi,df_rap,df_dance]

df_final = pd.concat(frames)




X= df_final.iloc[:,0:5]
y=df_final.iloc[:,6:]


X.describe()


df_final.corr()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
X


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=0.8,
        test_size=0.2,
        # random but same for all run, also accurancy depends on the
        # selection of data e.g. if we put 10 then accuracy will be 1.0
        # in this example
        random_state=23,
        # keep same proportion of 'target' in test and target data
        stratify=y
    )


from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=300)

xgb.fit(X_train, y_train)


print('The accuracy of the xgboost classifier is {:.2f} out of 1 on the test data'.format(xgb.score(X_test, y_test)))
preds = xgb.predict(X_test)
#acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
#acc=67%
#--------------------------------------------------------
from sklearn import preprocessing

loudness = df_final[['loudness']].values
min_max_scaler = preprocessing.MinMaxScaler()
loudness_scaled = min_max_scaler.fit_transform(loudness)
df_final['loudness'] = pd.DataFrame(loudness_scaled)


from sklearn.cluster import KMeans

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_final)
    Sum_of_squared_distances.append(km.inertia_)


kmeans = KMeans(n_clusters=3)
kmeans.fit(df_final)
#------------------------------------------------------
#PCA XG 
X= df_final.iloc[:,0:5]
y=df_final.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(
        principal_components, y,
        train_size=0.8,
        test_size=0.2,
        # random but same for all run, also accurancy depends on the
        # selection of data e.g. if we put 10 then accuracy will be 1.0
        # in this example
        random_state=23,
        # keep same proportion of 'target' in test and target data
        stratify=y
    )


xgb = XGBClassifier(n_estimators=150)

xgb.fit(X_train, y_train)

preds = xgb.predict(X_test)

acc_xgb_pca = (preds == y_test).sum().astype(float) / len(preds)*100
#-------------------------------------------------------


