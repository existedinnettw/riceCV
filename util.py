
from pathlib import Path, PurePath
import zipfile
from object_detection.utils import np_box_list, np_box_list_ops
import numpy as np


cur_path= Path().absolute()
tr_xs_dir= cur_path.parent.joinpath( 'data/Train_Dev/training' ) #may use match
tr_ys_dir= cur_path.parent.joinpath( 'data/Train_Dev/train_labels' ) #ground truth
tr_tfrecord_path= cur_path.parent.joinpath( 'data/riceOdTrain.tfrecord' )
tile_tr_tfrecord_path= tr_tfrecord_path.parent.joinpath('riceOdTileTrain.tfrecord')

eval_xs_dir= cur_path.parent.joinpath( PurePath( 'data/sample_data/ori_image') ) #may use match
eval_ys_dir= cur_path.parent.joinpath( PurePath( 'data/sample_data/csv_data') )
eval_tfrecord_path= cur_path.parent.joinpath( 'data/riceOdEval.tfrecord')
tile_eval_tfrecord_path= eval_tfrecord_path.parent.joinpath('riceOdTileEval.tfrecord')

dvlp_tfrecord_path= cur_path.parent.joinpath( 'data/riceOdDvlp.tfrecord')



def get_edge_tile_px(S=1728, T=8, OVLP_PCT=1/4):
    '''
    OVLP_PCT=1/4 #overlap percent of 1 edge of a tile
    S=1728 #2000x3000, 1728x2304, ori size
    T=8 #tile number
    ===
    return 
    WITH_OVLP_N: size of a tile without 1 OVLP_PART, used for per tile start px
    N: size of a tile
    '''
    N=S/((1-OVLP_PCT)*T+OVLP_PCT) #ideal case
    #print('ideal N:',N)
    #N=int(N)
    #OVLP_N= int(N*OVLP_PCT)

    WITH_OVLP_N=int((1-OVLP_PCT)*N)
    OVLP_N= S-WITH_OVLP_N*T
    N= WITH_OVLP_N+OVLP_N

    #print( WITH_OVLP_N, OVLP_N, N )
    #print('should same as ori size:',WITH_OVLP_N*T+OVLP_N)
    return WITH_OVLP_N, N

def get_jpg_paths_from_dir(tr_xs_dir):
    #tr_xs_fls=[ i.stem for i in tr_xs_dir.iterdir() ]
    #IMAGE_PATHS=[ str( tr_xs_dir.joinpath(i+'.jpg') ) for i in tr_xs_fls ]
    p = Path(tr_xs_dir)
    IMAGE_PATHS=list(p.glob('**/*.JPG'))
    return IMAGE_PATHS

def to_abs_path( sub_path, cur_path=None):
    if cur_path=='None':
        cur_path=Path().absolute()
    return cur_path.joinpath( sub_path )



def to_aidea_save(image, boxes, scores, target_path, thresh=0.3):
    '''
    image_path: pure string
    '''
    boxlist=np_box_list.BoxList(boxes) #[y_min, x_min, y_max, x_max]
    boxlist.data['scores']=scores
    boxlist=np_box_list_ops.filter_scores_greater_than(boxlist, thresh)
    detection_boxes=boxlist.data['boxes']
    
    im_height, im_width, _ = image.shape
    detection_boxes = detection_boxes * np.array([im_height, im_width, im_height, im_width])
    yc= (detection_boxes[:,0]+detection_boxes[:,2])/2 #round to int?
    xc= (detection_boxes[:,1]+detection_boxes[:,3])/2
    detection_centers= np.stack( (xc,yc), axis=1 ) # shape= (N, 2) (Cx,Cy)
    
    #print(target_path)
    np.savetxt( target_path , detection_centers, delimiter=",") #should be x,y
    return detection_boxes, detection_centers
    
# zip the inference csv file to 1 zipfile, which prepare for upload

def zip_dir(zip_path, dir_path, pattern):
    zf = zipfile.ZipFile( zip_path, mode='a' )
   
    for path_object in Path(dir_path).glob(pattern):
        #print(path_object, path_object.name )
        zf.write( path_object, arcname=path_object.name )
    zf.close()

def split_img(img,boxlist,T_ROW,T_COL,OVLP_PCT):
    '''
    img: image in numpy array
    
    return
    splited image 2d list, contain 3d ndarray as image
    boxlist, if input boxlist is not none
    '''
    h,w,c=img.shape
    sub_img_list=[]
    clipped_boxlist_list=[]
    # slice row
    for i in range(T_ROW):
        WITH_OVLP_N_ROW, N_ROW= get_edge_tile_px(S=h, T=T_ROW, OVLP_PCT=1/4)
        sub_img_list.append([])
        clipped_boxlist_list.append([])
        #slice col
        for j in range(T_COL):
            WITH_OVLP_N_COL, N_COL= get_edge_tile_px(S=w, T=T_COL, OVLP_PCT=1/4)
            b_y_min= WITH_OVLP_N_ROW*i
            b_y_max= b_y_min+N_ROW
            b_x_min= WITH_OVLP_N_COL*j
            b_x_max= b_x_min+N_COL
            sub_img=img[ b_y_min:b_y_max , b_x_min:b_x_max]
            sub_img_list[i].append(sub_img)
            #print(sub_img.shape)
            
            
            #chage box
            if boxlist!= None:
                window= [b_y_min/h, b_x_min/w, b_y_max/h, b_x_max/w] #normalized
                clipped_boxlist = np_box_list_ops.clip_to_window( boxlist=boxlist , window=window)
                #print('new slice {} {} boxes shape:{}'.format(i,j, clipped_boxlist.data['boxes'].shape) )
                clipped_boxlist= np_box_list_ops.change_coordinate_frame( boxlist=clipped_boxlist, window=window)
                clipped_boxlist_list[i].append(clipped_boxlist)
    return sub_img_list, clipped_boxlist_list

#def combine_img