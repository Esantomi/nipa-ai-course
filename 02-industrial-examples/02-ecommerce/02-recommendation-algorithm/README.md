# 추천 알고리즘 결과 확인
이번 실습에서는 **콘텐츠 기반의 추천 알고리즘을 사용하여 새로운 콘텐츠를 추천**받아 보도록 하겠습니다.

콘텐츠 기반의 추천 알고리즘은 콘텐츠와 콘텐츠 간의 유사성을 학습하여 특정 콘텐츠를 선택했을 때, 유사성이 높은 콘텐츠들을 리스트 업할 수 있습니다.

따라서 사용자의 **과거 시청 콘텐츠 데이터가 주어진다면 이 콘텐츠들과 유사한 콘텐츠들을 추천**할 수가 있습니다.

이번 실습에서는 넷플릭스의 영상 콘텐츠 데이터를 입력했을 때 유사성을 바탕으로 가장 비슷한 상위 10개의 추천 콘텐츠를 출력해 봅니다.

### 지시 사항
1. 우측 작은 따옴표 사이에 아래 예시 중 원하는 영화명을 골라 입력해 보세요.
    - `Vagabond`
    - `Pororo - The Little Penguin`
    - `The Lord of the Rings: The Return of the King`
    - `Larva`
2. `실행` 버튼을 눌러 출력된 결과를 살펴보세요.
3. `제출` 버튼을 눌러 추천 알고리즘의 결과가 올바르게 출력됐는지 확인해 보세요.

## 답안
```
import machine as ma

def main():
    
    netflix_overall, cosine_sim, indices = ma.preprocess()
    
    """
    지시사항 1번. 따옴표 사이에 들어가 있는 영화명을 지우고 왼쪽 지문의 예시 중 원하는 영화명을 입력해 보세요.
    """
    title = 'The Lord of the Rings: The Return of the King'
    
    print("{}와 비슷한 넷플릭스 콘텐츠를 추천합니다.".format(title))
    ma.get_recommendations_new(title, netflix_overall, cosine_sim, indices)


if __name__ == "__main__":
    main()
```

### 실행 결과
```
The Lord of the Rings: The Return of the King와 비슷한 넷플릭스 콘텐츠를 추천합니다.
                                                title  \
1               The Lord of the Rings: The Two Towers   
2                                    The Darkest Dawn   
3   Indiana Jones and the Kingdom of the Crystal S...   
4                                                   9   
5                                          The Matrix   
6                              The Matrix Revolutions   
7                                      V for Vendetta   
8                                         Singularity   
9                                 The Matrix Reloaded   
10                                          Supergirl   

                                   country  release_year  Similiarity  
1               New Zealand, United States          2002     0.808694  
2                           United Kingdom          2016     0.368605  
3                            United States          2008     0.287019  
4                            United States          2009     0.273009  
5                            United States          1999     0.273009  
6                            United States          2003     0.273009  
7   United States, United Kingdom, Germany          2005     0.266733  
8               Switzerland, United States          2017     0.260643  
9                            United States          2003     0.255377  
10           United Kingdom, United States          1984     0.252861  
```
