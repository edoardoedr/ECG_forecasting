class Estrattori_grafici_dati:

    def __init__(self, path):
        self.path = path  


    def get_graph(self, key,image):
        
            
        y_dim, x_dim, channel = image.shape
        x_values = [x for x in range(x_dim)]
        y_values = []
        
        for x in x_values:
            indexes = np.where(image[:,x,0] == 0)
            if indexes[0].size > 0:
                #print(indexes[0].max())
                y_values.append(y_dim - indexes[0].max())
            else:
                y_values.append(np.nan)

        return x_values,y_values    
    
    
    def interpolazione(self,x,y):
        
        x = np.array(x)
        y = np.array(y)
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask] 
        y = y[mask]
        # Matrice dei dati
        y_smooth = median_filter(y, size=3)

        data = np.vstack((x, y_smooth)).T

        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(data)

        labels = kmeans.labels_

        clean_cluster = 0 if np.sum(labels) > len(labels) / 2 else 1

        mask_clean = (labels == clean_cluster)

        plt.figure(figsize=(12, 6))
        plt.plot(x, y, 'o', label='Dati originali (con rumore)')
        # plt.plot(x, y_smooth, '-', label='Dati filtrati (Filtro Mediano)')
        # plt.plot(x[mask_clean], y_smooth[mask_clean], 'x', label='Dati puliti (Cluster)')
        print(len(x))
        plt.legend()
        plt.show()

                    



    
        
    def workflow(self,dizionario_immagini):
        
        for key,image in dizionario_immagini.items():
            x,y = self.get_graph(key,np.array(image))
            self.interpolazione(x,y)