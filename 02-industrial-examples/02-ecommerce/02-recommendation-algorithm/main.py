import machine as ma

def main():
    
    netflix_overall, cosine_sim, indices = ma.preprocess()
    
    """
    지시사항 1번. 따옴표 사이에 들어가 있는 영화명을 지우고 왼쪽 지문의 예시 중 원하는 영화명을 입력해보세요.
    """
    title = 'The Lord of the Rings: The Return of the King'
    
    print("{}와 비슷한 넷플릭스 콘텐츠를 추천합니다.".format(title))
    ma.get_recommendations_new(title, netflix_overall, cosine_sim, indices)


if __name__ == "__main__":
    main()
