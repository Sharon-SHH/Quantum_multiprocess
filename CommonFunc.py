import numpy as np
import os
from datetime import datetime

class CommonFunc(object):
    def loadList(self, save_name):
        # the filename should mention the extension 'npy'
        tempNumpyArray = np.load(save_name)
        print('Loading...', save_name)
        return tempNumpyArray.tolist()

    def saveList(self, myList, max_clique_size='max_clique_size.npy'):
        np.save(max_clique_size, myList)
        print(max_clique_size, "Saved successfully!")

    def save_to_file(self, path, cliq_size_p, clique_symbols):
        save_cliques_file = path + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ".txt"
        mode = 'a' if os.path.exists(save_cliques_file) else 'w'

        with open(save_cliques_file, mode) as result_file:
            result_file.write(str(cliq_size_p) + "\n")
            result_file.write(str(clique_symbols) + "\n")
        result_file.close()
        return save_cliques_file