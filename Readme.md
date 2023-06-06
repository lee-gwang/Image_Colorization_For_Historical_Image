# 흑백 이미지 복원을 위한 컬러화 기술 개발
<p align="center"><img src="./example/many_people_nature.gif"></p>

5.18 민주화 운동 당시의 역사적 의미를 가지는 흑백이미지를 딥러닝 기술을 활용해 컬러화 시키는 프로젝트입니다. 본 프로젝트는 SK텔레콤에서 AI Fellowship 학생으로 선정되고 지원을 받아 개발된 코드입니다. 코드는 일부분만 공개되었으며, 더욱 고도화된 기술을 원하시는 분은 따로 메일로 문의주세요.

코드는 크게 Automatic Colorization, Hint-based Colorization, Multi-modal Colorization으로 이루어져있습니다. Multi-modal Colrozation 코드를 통해 카카오채널에서 흑백이미지를 복원하는 챗봇까지 개발했습니다.

This is a project that utilizes deep learning technology to colorize black-and-white images with historical significance during the Gwangju democratization movement of May 18th. The project was developed with the support of SK Telecom as part of the AI Fellowship. Only a portion of the code has been made public, and if you are interested in more advanced techniques, please inquire via email. (icanlgh0016@gmail.com / toqufns@gmail.com)

The code consists of three main parts: Automatic Colorization, Hint-based Colorization, and Multi-modal Colorization.
- We developed Demo for Automatic Colorization and Hint-based Colorization [[DEMO]](https://github.com/SaebyeolShin/Colorization_UI)
- Through the Multi-modal Colorization code, a Chatbot has been developed to restore black-and-white images on the Kakao Channel. [[KAKAO CHANNEL]](http://pf.kakao.com/_mxgELxj)


# Automatic Colorization
<p align="center"><img src="./example/vis1.gif"></p>

"Automatic Colorization"은 유저의 가이드 없이 예측하는 방법입니다. 해당 방법은 역사적 사실과 맞지 않은 이미지 채색이 될 수가있었기에, 새로운 방법이 필요했습니다.

"Automatic Colorization" is a method that predicts colors without user guidance. However, this method had the potential to colorize images in a way that did not align with historical facts, necessitating the development of a new approach.

# Hint-based Colorization
<p align="center"><img src="./example/model.png"></p>
<p align="center"><img src="./example/video1.gif"></p>

"Hint-based Colorization"은 유저의 가이드에 따라 유저가 원하는 색으로 이미지 채색이 진행됩니다. 따라서 역사적 의미를 가지는 사진에 유저의 사전지식을 사용해 이미치 채색을 가능하게 만듭니다.

"Hint-based Colorization" involves coloring images according to user guidance, allowing users to specify the desired colors for the image. This enables colorization of images with historical significance using the user's prior knowledge.

# Multi-modal Colorization (Text-based Colorization)
<p align="center"><img src="./example/multi.jpg"></p>

해당 방법을 개발하게 된 계기는 플랫폼에 따라 생기는 기술적용 문제점 때문입니다. 유저가 상호작용 가능한 "Image Colorization" 기법을 배포하기 위해서는 어떤 플랫폼에서 사용하느냐에 따라 기술적용이 많이 달라졌습니다. 저는 카카오챗봇을 통해 기술을 적용하려 했지만, 일반적으로 마우스 클릭을 통해 유저 가이드를 제공하는 방식은 카카오톡에서 사용하기 어려웠습니다.
이에따라, 카카오톡에서 "Text"만으로도 이미지 채색이 가능하도록 하는 방법을 개발했습니다. 위 사진에서는 중간과정이 생략되었지만, CLIP을 통해 원하는 영역을 찾고 해당 하는 영역에 유저 가이드를 제공하여 "텍스트 기반 이미지 채색"이 가능하도록 만들었습니다.

The motivation behind developing this method was the technical implementation challenges posed by different platforms. The application of the "Image Colorization" technique, which allows user interaction, varied significantly depending on the platform. While we attempted to apply the technique through the Kakao chatbot, providing user guidance through mouse clicks, it was challenging to do so in KakaoTalk.
Therefore, we developed a method that enables image colorization in KakaoTalk using only "Text" as input. In the image provided, intermediate steps have been omitted, but the method uses CLIP to identify the desired regions and provides user guidance for those regions, allowing for "text-based image colorization."


# 후속 프로젝트
## Image Color Transfer
<p align="center"><img src="./example/transfer1.png"></p>
<p align="center"><img src="./example/transfer2.png"></p>
<p align="center"><img src="./example/transfer3.png"></p>
<p align="center"><img src="./example/transfer4.png"></p>

단순히 흑백이미지를 복원하는 것이 아니라, 해당 기술을 활용해 컬러 이미지의 일부 영역을 다른 색으로 변환할 수 있지 않을까? 하는 의문에서 해당 프로젝트를 시작했습니다. 다른 색으로 "Recolorization" 하기 위해서는 유저의 가이드만 인풋으로 함께 입력하면 됩니다. 현재는 나이브하게 코드가 개발되어 있고, 학습 자체에 많은 이슈가 존재하여 추후에 이를 해결할 생각입니다.

Rather than simply restoring black-and-white images, we wondered if this technology could be used to convert certain regions of color images into different colors. To achieve "Recolorization" into different colors, only the user's guidance needs to be provided as input. Currently, the code is naively developed, and there are many issues regarding the training process that need to be addressed, which we plan to resolve in the future.

## 6.25 참전 용사 이미지 복원
<p align="center"><img src="./example/625_1.PNG"></p>
<p align="center"><img src="./example/625_2.PNG"></p>

SK텔레콤에서의 인턴을 마친 후, 조금 더 의미있는 곳에 기술을 적용해보고자 6.25 참전용사 분들의 이미지를 복원해보았습니다. 옛날 사진들에 노이즈가 너무 많아 추가적인 학습 테크닉이 필요했습니다. 본 코드에는 따로 공개하지 않았습니다.

After completing an internship at SK Telecom, we wanted to apply the technology to a more meaningful way, so we attempted to restore images of the veterans in the Korean War. Since the old photos had a lot of noise, additional training techniques were required, but we have not publicly released this specific code in this page.

# Thanks for
- iColoriT (https://github.com/pmh9960/iColoriT)

    ViT를 사용해 Hint-based Colorization 코드가 깔끔하게 개되어 있어 정말 많은 도움을 받았습니다, 또한 Demo를 만드는데 많은 도움을 받았습니다.

    Our GUI is an updated version of the iColoriT: Towards Propagating Local Hint to the Right Region in Interactive Colorization by Leveraging Vision Transformer. Thanks for sharing the codes!


# TODO
- 각각 코드에 학습 방법 readme 작성하기
