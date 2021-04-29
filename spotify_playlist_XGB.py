import spotipy
import pandas as pd
import seaborn as sns
import spotipy.oauth2 as oauth2

def generate_token():
    """ Generate the token. Please respect these credentials :) """
    credentials = oauth2.SpotifyClientCredentials(
        client_id='5e22c02862e0444aac0351ee2aca9exx',
        client_secret='dd2407b5b3984fa38bf707b883e34bxx')
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
        
        client_credentials_manager = oauth2.SpotifyClientCredentials(client_id='5e22c02862e0444aac0351ee2aca9exx', client_secret='dd2407b5b3984fa38bf707b883e34bxx')
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
    df_rock1 = write_tracks(text_file, tracks)
    return df_rock1


token = generate_token()
spotify = spotipy.Spotify(auth=token)
    
#sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# example playlist
df_rock1 = write_playlist('abc', '3Ho3iO0iJykgEQNbjB2sic')


#run above code for each playlist of different genres 
df_final.to_csv(r'D:\Proj\df_final.csv', index=False)

#Combine playlists
frames_rock = [df_rock2,df_rock1]
df_rock_final = pd.concat(frames_rock)

frames_lofi = [df_lofi2,df_lofi1]
df_lofi_final = pd.concat(frames_lofi)

#Assign genres
df_lofi_final['type']=1
df_bolly['type']=2
df_rock_final['type']=3

#Combine df

frames = [df_lofi_final,df_bolly,df_rock_final]
df_final = pd.concat(frames)

'''
df_rock_final.to_csv(r'D:\Proj\rock.csv', index=False)

frames_lofi = [df_lofi2,df_lofi1]
df_lofi_final = pd.concat(frames_lofi)

frames_rock = [df_rock2,df_rock1]
df_rock_final = pd.concat(frames_rock)

df_lofi1.to_csv(r'D:\Proj\lofi2.csv', index=False)
df_lofi2.to_csv(r'D:\Proj\lofi1.csv', index=False)
frames_lofi = [df_lofi2,df_lofi1]
df_lofi_final = pd.concat(frames_lofi)


df_bolly.to_csv(r'D:\Proj\bolly.csv', index=False)

df_rock_final.to_csv(r'D:\Proj\rock.csv', index=False)
frames_rock = [df_rock2,df_rock1]
df_rock_final = pd.concat(frames_rock)


df_lofi1 = pd.read_csv (r'D:\Proj\lofi2.csv')
df_lofi2 = pd.read_csv (r'D:\Proj\lofi1.csv')
df_rock_final = pd.read_csv (r'D:\Proj\rock.csv')
df_bolly = pd.read_csv (r'D:\Proj\bolly.csv')

df_lofi_final['type']=1
df_bolly['type']=2
df_rock_final['type']=3

frames = [df_lofi_final,df_bolly,df_rock_final]
#frames = [df_lofi_final,df_bolly]
df_final = pd.concat(frames)

from sklearn import preprocessing




loudness = df_final[['loudness']].values
min_max_scaler = preprocessing.MinMaxScaler()
loudness_scaled = min_max_scaler.fit_transform(loudness)
df_final['loudness'] = pd.DataFrame(loudness_scaled)
'''


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


#Save model
import pickle
pickle.dump(model, open(r'D:\Proj\spotify_model.pkl', 'wb'))


#model2 = pickle.load(open(r'D:\Proj\spotify_model.pkl', 'rb'))


'''
#Test song individually

client_credentials_manager = oauth2.SpotifyClientCredentials(client_id='5e22c02862e0444aac0351ee2aca9e2b', client_secret='dd2407b5b3984fa38bf707b883e34bea')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

aud_features  = sp.audio_features('2ehrRaLsjpqkQNZUAHGs78')

danceability =aud_features[0]['danceability']
loudness = aud_features[0]['loudness']
speechiness =aud_features[0]['speechiness']
acousticness = aud_features[0]['acousticness']
liveness =aud_features[0]['liveness']
instrumentalness =aud_features[0]['instrumentalness']



print(model2.predict([[danceability,loudness,speechiness,acousticness,liveness,instrumentalness]]))



'''




