import cv2
import numpy as np
import os

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 2)
    img_canny = cv2.Canny(img_blur, 10, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def put_text_in_image(img, text, x, y):
    img = cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA, False)
    return img

def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1: j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]): return tuple(points[j])

def extract_layers(img, cache_path):
    h,w,_ = img.shape
    img = cv2.resize(img, (w//4,h//4), interpolation = cv2.INTER_AREA)
    contours, hierarchy = cv2.findContours(preprocess(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    arrows, boxes = [], []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)

        area = cv2.contourArea(cnt)
        if area<500:continue
        # cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
            
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        if sides == 4 and len(approx[:,0,:])==4:
            
            boxes.append({
                    "x1":x, 
                    "x2":x+w, 
                    "y1": y, 
                    "y2":y+h, 
                    "connected_arrows":[],
                    "layer type":""
                }
            )
        elif 6 > sides > 3 and sides + 2 == len(approx):
            arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
            if arrow_tip:
                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
                cv2.circle(img, arrow_tip, 3, (0, 0, 255), cv2.FILLED)

                arrows.append(
                    {
                        "tip":arrow_tip,
                        "x1":x, 
                        "x2":x+w, 
                        "y1": y, 
                        "y2":y+h,
                        "left": h<w and abs((x+w)-arrow_tip[0]) > abs(x-arrow_tip[0]),
                        "right": h<w and abs((x+w)-arrow_tip[0]) < abs(x-arrow_tip[0]),
                        "up": h>w and abs((y+h)-arrow_tip[1]) > abs(y-arrow_tip[1]),
                        "down": h>w and abs((y+h)-arrow_tip[1]) < abs(y-arrow_tip[1]),
                    }
                )

    for i in range(len(arrows)):
        a = arrows[i]
        tip = a["tip"]

        for j in range(len(boxes)):
            if (a["up"] or a["down"]) and (boxes[j]["x1"] < tip[0] < boxes[j]["x2"]):
                boxes[j]["connected_arrows"].append((i,"vert",a["up"]))
            if (a["left"] or a["right"]) and (boxes[j]["y1"] < tip[1] < boxes[j]["y2"]):
                boxes[j]["connected_arrows"].append((i,"hor",a["left"]))

    curr_arrow_index = 0
    for b in boxes:
        connected_arrows = b["connected_arrows"]
        if len(connected_arrows) == 1:
            arrow_index = connected_arrows[0][0]
            if connected_arrows[0][1] == 'vert':
                if (arrows[arrow_index]["y1"] < b["y1"] and connected_arrows[0][2]) or \
                    (arrows[arrow_index]["y1"] > b["y1"] and not connected_arrows[0][2]):
                    img = put_text_in_image(img,"input",b["x1"],b["y1"])
                    b["layer type"] = "input"
                else:
                    img = put_text_in_image(img,"output",b["x1"],b["y1"])
                    b["layer type"] = "output"
            else:
                if (arrows[arrow_index]["x1"] < b["x1"] and connected_arrows[0][2]) or \
                    (arrows[arrow_index]["x1"] > b["x1"] and not connected_arrows[0][2]):
                    img = put_text_in_image(img,"input",b["x1"],b["y1"])
                    b["layer type"] = "input"
                else:
                    img = put_text_in_image(img,"output",b["x1"],b["y1"])
                    b["layer type"] = "output"
            
            if b["layer type"] == "input" : curr_arrow_index = connected_arrows[0][0]
            cv2.imwrite(os.path.join(cache_path, b["layer type"] + ".png"), img[b["y1"]:b["y2"],b["x1"]:b["x2"]])

    hidden_count = 1
    for i in range(len(boxes)):
        for b in boxes:
            connected_arrows = b["connected_arrows"]
            
            if b["layer type"] != "":continue
            try:
                if connected_arrows[0][0] == curr_arrow_index or connected_arrows[1][0] == curr_arrow_index:
                    curr_arrow_index = connected_arrows[1][0] if curr_arrow_index==connected_arrows[0][0] else connected_arrows[0][0]
                    img = put_text_in_image(img, f"hidden{hidden_count}",b["x1"],b["y1"])
                    b["layer type"] = f"hidden{hidden_count}"
                    hidden_count+=1

                    cv2.imwrite(os.path.join(cache_path, b["layer type"] + ".png"), img[b["y1"]:b["y2"],b["x1"]:b["x2"]])
            except:
                pass
    
    cv2.imwrite(os.path.join(cache_path, "res.png"), img)
    return cache_path