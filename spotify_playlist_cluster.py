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
df_rap = write_playlist('abc', '2G2zkK3cBVXB1jSsDMQSk3')


#run above code for each playlist of different genres 

#run below code to combine each data frame of genres 

frames = [df_lofi,df_rap,df_pop]

df_final = pd.concat(frames)

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

from sklearn.decomposition import PCA
y_kmeans = kmeans.predict(df_final)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_final)



pc = pd.DataFrame(principal_components)
pc['label'] = y_kmeans
pc.columns = ['x', 'y','label']

#plot data with seaborn
cluster = sns.lmplot(data=pc, x='x', y='y', hue='label', 
                   fit_reg=False, legend=True, legend_out=True)



df_final['label'] = y_kmeans

# shuffle dataset

df_final = df_final.sample(frac=1)
df_final['label'].value_counts()

y=df_final.label
X=df_final.drop('label',axis=1)




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

rfc = RandomForestClassifier(n_estimators=100,criterion='gini')
rfc.fit(X_train,y_train)


y_pred = rfc.predict(X_test)
# Training model using Naive bayes classifier


from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)

print(confusion_m)

# Saving model to disk
import pickle
pickle.dump(rfc, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0.676, 0.940,0.228,0.0237,0.19,0.00348]]))
