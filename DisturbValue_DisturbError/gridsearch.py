
def grid(grid_on, dv_annealing):

    if (grid_on[0] == 'y') and (grid_on[1] == 'n') and (grid_on[2] == 'n') and (grid_on[3] == 'n'):
        grid_search1 = [0]
        grid_search2 = [.05, .1, .3, .4, .5]
        switch = 0
    
    elif (grid_on[0] == 'n') and (grid_on[1] == 'y') and (grid_on[2] == 'n') and (grid_on[3] == 'n'):    
        grid_search1 = [0]
        if dv_annealing == 'y':
            grid_search2 = [0]#[0.05, 0.07, 0.12, 0.1, 0.15, .2, .25, .3]
        else: 
            grid_search2 = [0.05, 0.07, 0.12, 0.1, 0.15, .2, .25, .3]
        switch = 1
        
    elif (grid_on[0] == 'n') and (grid_on[1] == 'n') and (grid_on[2] == 'y') and (grid_on[3] == 'n'):  
        grid_search1 = [0]
        grid_search2 = [2, 4, 6, 8]
        switch = 2
        
    elif (grid_on[0] == 'n') and (grid_on[1] == 'n') and (grid_on[2] == 'n') and (grid_on[3] == 'y'):  
        grid_search1 = [0]
        grid_search2 = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3] 
        
        switch = 3
        
    elif (grid_on[0] == 'y') and (grid_on[1] == 'y') and (grid_on[2] == 'n') and (grid_on[3] == 'n'):  
        grid_search1 = [.05, .1, .3, .4, .5]
        grid_search2 = [0.05, 0.07, 0.1, 0.12, 0.15, .2, .25, .3]
        switch = 4  
        
    elif (grid_on[0] == 'y') and (grid_on[1] == 'n') and (grid_on[2] == 'y') and (grid_on[3] == 'n'):  
        grid_search1 = [.05, .1, .3, .4, .5]
        grid_search2 = [2, 4, 6, 8]
        switch = 5    
    
    elif (grid_on[0] == 'y') and (grid_on[1] == 'n') and (grid_on[2] == 'n') and (grid_on[3] == 'y'):  
        grid_search1 = [.05, .1, .3, .4, .5]
        grid_search2 = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3] 
        switch = 6    
    
    elif (grid_on[0] == 'n') and (grid_on[1] == 'y') and (grid_on[2] == 'y') and (grid_on[3] == 'n'):  
        grid_search1 = [0.05, 0.07, 0.12, 0.1, 0.15, .2, .25, .3]
        grid_search2 = [2, 4, 6, 8]
        switch = 7 
        
    elif (grid_on[0] == 'n') and (grid_on[1] == 'y') and (grid_on[2] == 'n') and (grid_on[3] == 'y'):  
        grid_search1 = [0.05, 0.07, 0.12, 0.1, 0.15, .2, .25, .3]
        grid_search2 = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3] 
        switch = 8   
        
    elif (grid_on[0] == 'n') and (grid_on[1] == 'n') and (grid_on[2] == 'y') and (grid_on[3] == 'y'):  
        grid_search1 = [2, 4, 6, 8]
        grid_search2 = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3] 
        switch = 9  
    
    elif (grid_on[0] == 'n') and (grid_on[1] == 'n') and (grid_on[2] == 'n') and (grid_on[3] == 'n'):     
        grid_search1 = [0]
        grid_search2 = [0]
        switch = 10
        
    else:
        print("Your option is not a single mode.")

    return grid_search1, grid_search2, switch     
