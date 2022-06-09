import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns
import os 
sns.set()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', default = 'pong',
                        type = str, help = 'pong, breakout, pacman, montezuma')
    parser.add_argument('-s', '--score', default = True,
                        help = 'show experiments\' score')
                        
    args = parser.parse_args()
    
    env_path = 'results/' + args.environment + '/'
    experiments = os.listdir(env_path)
    
    if args.score == True:
        s = '_score.pdf'
        fig, ax = plt.subplots(2, 1, figsize=(15,10))
        
        i = 0
        for name in experiments:
            path = env_path + name
            if os.path.isdir(path):
                print(name)
                data = pd.read_csv(path + '/logs_score.csv', sep = ',')
                ax[0].plot(data['iteration'], data['avg_score'], label = name)
                ax[1].plot(data['iteration'], data['best_avg_score'], label = name)
                i += 1
        
        ax[0].set_title('Average score')
        ax[0].set_xlabel('number of iterations')
        ax[0].set_ylabel('score')
        ax[0].legend()
        
        ax[1].set_title('Best average score')
        ax[1].set_xlabel('number of iterations')
        ax[1].set_ylabel('score')
    else:
        s = '_loss.pdf'
        fig, ax = plt.subplots(3, 1, figsize=(18,10))

        i = 0
        for name in experiments:
            path = env_path + name
            if os.path.isdir(path):
                print(name)
                data = pd.read_csv(path + '/logs_loss.csv', sep = ',')
                ax[0].plot(data['iteration'], data['policy'], label = name)
                if 'value' in data.columns:
                    ax[1].plot(data['iteration'], data['value'], label = name)
                else:
                    ax[1].plot(data['iteration'], data['ext_value'], label = name)
                ax[2].plot(data['iteration'], data['entropy'], label = name)
                i += 1
        
        ax[0].set_title('Policy loss')
        ax[0].set_ylabel('loss')
        ax[0].legend()
        
        ax[1].set_title('Value loss')
        ax[1].set_ylabel('loss')
        
        
        ax[2].set_title('Entropy loss')
        ax[2].set_xlabel('number of iterations')
        ax[2].set_ylabel('loss')
    
    
    fig.savefig('plots/' + args.environment + s, dpi='figure')

