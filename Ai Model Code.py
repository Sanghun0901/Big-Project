from keras.models import load_model
import cv2
import numpy as np
import datetime
import sys

sys.stdout = open(r'C:\B_data\output.txt','w')

t_cnt = 0
f_cnt = 0
LED = 'OFF'
class_name = 0

np.set_printoptions(suppress=True) # 부동소수점 표기 금지

model = load_model(r"C:\B_data\converted_keras\keras_model_26.h5", compile=False) # 모델 경로 설정

class_names = open(r"C:\B_data\converted_keras\labels.txt", 'rt', encoding='UTF8').readlines() # 라벨 경로 설정

video = cv2.VideoCapture(r'C:\B_data\test\test_칠번.mp4') # 테스트 영상 경로 설정

fps = video.get(cv2.CAP_PROP_FPS) # FPS 산출
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) # 영상 가로값 산출
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 영상 세로값 산출

output_video = cv2.VideoWriter('C:\B_data\output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) # 저장용 영상 설정(경로, 코덱, fps, (가로, 세로)

test_wa = cv2.imread(r'C:\Users\User\Desktop\빅프로젝트\3주차\그림2.png')

while True: 
    if class_name == '0 True\n':
        t_cnt += 1 
        if t_cnt >= 3 :
            f_cnt = 0
            LED = 'ON'
    elif class_name == '1 False\n' :
        f_cnt += 1
        if f_cnt >= 3:
            t_cnt = 0
            LED = 'OFF'
            
    else : 
        pass
    
    
    ret, image = video.read() # 비디오 프레임 단위로 나눠서 불러오기

    if not ret: # 프레임이 없는 경우 멈춤
        break

    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4) # 프레임 크기 조절

    input_image = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3) # 넘파이 형태로 변형
    input_image = (input_image / 127.5) - 1

    prediction = model.predict(input_image) # 모델로 해당 이미지의 상태를 예측

    True_LED = cv2.imread(r'C:\B_data\True_LED.jpg')
    False_LED = cv2.imread(r'C:\B_data\False_LED.jpg')
    War_LED = cv2.imread(r'C:\B_data\War_LED.jpg')

    True_LED = cv2.resize(True_LED, (150,350), interpolation=cv2.INTER_LANCZOS4)
    False_LED = cv2.resize(False_LED, (150,350), interpolation=cv2.INTER_LANCZOS4)
    War_LED = cv2.resize(War_LED, (150,350), interpolation=cv2.INTER_LANCZOS4)

    if np.argmax(prediction) == 0 : 
        class_label = class_names[0].strip()
        test_wa = cv2.imread(r'C:\B_data\True.png')
        True_LED = cv2.imread(r'C:\B_data\True_LED.jpg')
        
        test_img =image
        test_wa = cv2.resize(test_wa, (250,125), interpolation=cv2.INTER_LANCZOS4)
        True_LED = cv2.resize(True_LED, (150,350), interpolation=cv2.INTER_LANCZOS4)
        
        rows, cols,channels = test_wa.shape
        
        start_row = 50
        start_col = 1470
        end_row = start_row + rows
        end_col = start_col + cols
        
        rows1, cols1,channels1 = True_LED.shape
        
        start_row1 = 50
        start_col1 = 1750
        end_row1 = start_row1 + rows1
        end_col1 = start_col1 + cols1

        test_img[start_row:end_row, start_col:end_col] = test_wa
        
        if t_cnt > 4 :
            test_img[start_row1:end_row1, start_col1:end_col1] = True_LED
        else :
            rows1, cols1,channels1 = False_LED.shape
        
            start_row1 = 50
            start_col1 = 1750
            end_row1 = start_row1 + rows1
            end_col1 = start_col1 + cols1
            test_img[start_row1:end_row1, start_col1:end_col1] = False_LED
        
        output_video.write(test_img) # 저장용 영상에 이미지 삽입 및 저장

        
    elif np.argmax(prediction) == 1 : 
        class_label = class_names[1].strip()
        test_wa = cv2.imread(r'C:\B_data\False.png')
        False_LED = cv2.imread(r'C:\B_data\False_LED.jpg')
    
        
        test_img =image
        test_wa = cv2.resize(test_wa, (250,125), interpolation=cv2.INTER_LANCZOS4)
        False_LED = cv2.resize(False_LED, (150,350), interpolation=cv2.INTER_LANCZOS4)
        
        rows, cols,channels = test_wa.shape
        
        start_row = 50
        start_col = 1470
        end_row = start_row + rows
        end_col = start_col + cols
        
        rows1, cols1,channels1 = False_LED.shape
        
        start_row1 = 50
        start_col1 = 1750
        end_row1 = start_row1 + rows1
        end_col1 = start_col1 + cols1

        test_img[start_row:end_row, start_col:end_col] = test_wa
        test_img[start_row1:end_row1, start_col1:end_col1] = False_LED


        output_video.write(test_img) # 저장용 영상에 이미지 삽입 및 저장
    
    elif np.argmax(prediction) == 2 : 
        class_label = class_names[2].strip()
        test_wa = cv2.imread(r'C:\B_data\war.png')
        True_LED = cv2.imread(r'C:\B_data\True_LED.jpg')

        test_img =image
        
        test_wa = cv2.resize(test_wa, (250,125), interpolation=cv2.INTER_LANCZOS4)
        True_LED = cv2.resize(True_LED, (150,350), interpolation=cv2.INTER_LANCZOS4)
        
        rows, cols,channels = test_wa.shape
        
        start_row = 50
        start_col = 1470
        end_row = start_row + rows
        end_col = start_col + cols
        
        rows1, cols1,channels1 = True_LED.shape
        
        start_row1 = 50
        start_col1 = 1750
        end_row1 = start_row1 + rows1
        end_col1 = start_col1 + cols1
        
        test_img[start_row:end_row, start_col:end_col] = test_wa
        if LED =='ON':
            test_img[start_row1:end_row1, start_col1:end_col1] = False_LED

        cv2.putText(test_img, str(datetime.datetime.now().strftime('%m/%d %H:%M:%S')) , (300, 275), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255), 3, cv2.LINE_AA,bottomLeftOrigin=False)
        output_video.write(test_img) # 저장용 영상에 이미지 삽입 및 저장
    
    else : 
        class_label = class_names[2].strip()
        test_wa = cv2.imread(r'C:\B_data\anly.png')
        test_img =image
        
        test_wa = cv2.resize(test_wa, (250,125), interpolation=cv2.INTER_LANCZOS4)

        rows, cols,channels = test_wa.shape
        
        start_row = 50
        start_col = 1470
        end_row = start_row + rows
        end_col = start_col + cols

        test_img[start_row:end_row, start_col:end_col] = test_wa

        if LED =='ON'and t_cnt > 4:      
            rows1, cols1,channels1 = True_LED.shape
            
            start_row1 = 50
            start_col1 = 1750
            end_row1 = start_row1 + rows1
            end_col1 = start_col1 + cols1
            test_img[start_row1:end_row1, start_col1:end_col1] = True_LED
            
        if LED =='OFF' :
            rows1, cols1,channels1 = False_LED.shape
            
            start_row1 = 50
            start_col1 = 1750
            end_row1 = start_row1 + rows1
            end_col1 = start_col1 + cols1
            test_img[start_row1:end_row1, start_col1:end_col1] = False_LED
            
        else :
            rows1, cols1,channels1 = True_LED.shape
            
            start_row1 = 50
            start_col1 = 1750
            end_row1 = start_row1 + rows1
            end_col1 = start_col1 + cols1
            test_img[start_row1:end_row1, start_col1:end_col1] = True_LED
        
        output_video.write(image)

    cv2.imshow('Walk Monitoring', image) # 확인용 영상 틀기

    if cv2.waitKey(1) == ord('q'): # q가 입력되면 작업 종료
        break

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    output_video.write(image)
    
    if class_name[2:] == 'Warning\n' : 
        open(r'C:\B_data\output.txt','a')
        print("상태: ", '보행자 존재 ', end="")
        print(" / 정확도 :", str(round(confidence_score * 100,2)), "%", end="")
        print('LED :',LED)
    else : 
        if class_name[2:] == 'True\n' :
            print("상태: ", '보행자 존재 ', end="")
            print(" / 정확도 :", str(round(confidence_score * 100,2)), "%", end="")
            print('LED :',LED)
        elif class_name[2:] == 'False\n' :
            print("상태:", '보행자 없음 ', end="")
            print(" / 정확도 :", str(round(confidence_score * 100,2)), "%", end="")
            print('LED :',LED)

    keyboard_input = cv2.waitKey(1)

    
    if keyboard_input == 27: # 27이 입력되면 작업 종료q
        break
    

         
output_video.release() # 저장용 영상 최종 종료 및 저장 완료
video.release() # 비디오 작업 종료
cv2.destroyAllWindows() # 모든 작업 종료 및 호출 창 닫기
