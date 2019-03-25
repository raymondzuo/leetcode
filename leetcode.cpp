#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <set>
#include <stack>

using std::cout;
using std::cin;
using std::endl;
using std::vector;
using std::map;
using std::queue;
using std::pair;
using std::string;
using std::max;
using std::set;
using std::stack;

/**
 * leetcode most frequency problems
 * 
 */

//#1 Two sum
vector<int> twoSum(vector<int>& nums, int target)
{
    vector<int> result;
    std::map<int, int> mymap;

    for(int i = 0; i < nums.size(); i++)    
    {
        int another = target - nums[i];
        auto it = mymap.find(another);
        if(it != mymap.end())
        {
            result.push_back(it->second);
            result.push_back(i);
            return result;
        }

        mymap[nums[i]] = i;
    }
}

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
//#2 Add two numbers 
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2)
{
    ListNode *head = new ListNode(0);
    ListNode *cur = head;
    int carry = 0;
    ListNode *p1 = l1, *p2 = l2;

    while(p1 || p2)
    {
        int x = p1 ? p1->val : 0;
        int y = p2 ? p2->val : 0;

        int sum = x + y + carry;
        carry = sum / 10;
        sum = sum % 10;
        ListNode *temp = new ListNode(sum); 

        cur->next = temp;
        cur = temp;

        if(p1)
            p1 = p1->next;
        if(p2)
            p2 = p2->next;
    }

    if(carry > 0)
        cur->next = new ListNode(carry);

    return head->next;
}

/********************/
//#200. Number of Islands
void dfs(vector<vector<char>>& grid, int r, int c)
{
    int maxr = grid.size();
    int maxc = grid[0].size();

    if(r < 0 || c < 0 || r >= grid.size() || c >= grid[0].size() || grid[r][c] == '0')
        return;
    grid[r][c] = '0';
    dfs(grid, r + 1, c);
    dfs(grid, r - 1, c);
    dfs(grid, r, c + 1);
    dfs(grid, r, c - 1);
}

int numIslands(vector<vector<char>>& grid) 
{
    int count = 0;

    int maxr = grid.size();
    if(0 == maxr)
        return count;
    int maxc = grid[0].size();
    if(0 == maxc)
        return count;

    for(int i = 0; i < maxr; i++)
    {
        for(int j = 0; j < maxc; j++)
        {
            if('1' == grid[i][j]) 
            {
                count++;
                dfs(grid, i, j);
            }
        }
    }

    return count;
}

int numIslandsBFS(vector<vector<char>>& grid) 
{
    int nr = grid.size();
    if(0 == nr)
        return 0;
    int nc = grid[0].size();

    int count = 0;
    for(int r = 0; r < nr; r++)
    {
        for(int c = 0; c < nc; c++)
        {
            if('1' == grid[r][c])
            {
                count++;
                grid[r][c] = '0';
                queue<pair<int, int>> tempqueue;                
                tempqueue.push({r,c});

                while(!tempqueue.empty())
                {
                    auto front = tempqueue.front();
                    tempqueue.pop();
                    int tempr = front.first;
                    int tempc = front.second;
                    
                    if(tempr - 1 >= 0 && grid[tempr - 1][tempc] == '1')
                    {
                        tempqueue.push({tempr - 1, tempc});
                        grid[tempr - 1][tempc] = '0';
                    }
                    if(tempr + 1 < nr && grid[tempr + 1][tempc] == '1')
                    {
                        tempqueue.push({tempr + 1, tempc});
                        grid[tempr + 1][tempc] = '0';
                    }
                    if(tempc - 1 >= 0 && grid[tempr][tempc - 1] == '1')
                    {
                        tempqueue.push({tempr, tempc - 1});
                        grid[tempr][tempc - 1] = '0';
                    }
                    if(tempc + 1 < nc && grid[tempr][tempc + 1] == '1')
                    {
                        tempqueue.push({tempr, tempc + 1});
                        grid[tempr][tempc + 1] = '0';
                    }
                }
            }
        }
    }

    return count;
}

/********************/

/********************/
//#5. Longest Palindromic Substring
string longestPalindrome(string s)
{
    int n = s.length(); 
    int max = 1;
    int si = 0;//start
    int dp[n][n];//n->n from n to n
    //length is 1
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(i == j)
                dp[i][j] = 1;
            else
                dp[i][j] = 0;
        }
    }
    //length is 2
    for(int i = 0; i < n - 1; i++)
    {
        if(s[i + 1] == s[i])
        {
            dp[i][i + 1] = 1;
            max = 2;
            si = i;
        }
    }

    //length large than 3
    for(int k = 3; k <= n; k++)
    {
        for(int i = 0; i < n - k + 1; i++)
        {
            int j = i + k - 1;
            if(dp[i + 1][j - 1] && s[i] == s[j])
            {
                dp[i][j] = 1;
                if(k > max)
                {
                    max = k;
                    si = i;
                }
            }
        }
    }

    return s.substr(si, max);
}

/********************/
/* 3. Longest Substring Without Repeating Characters*/
int lengthOfLongestSubstring(string s)
{
    int max = 0, start = 0;
    vector<char> dp(256, -1);

    for(int i = 0; i < s.length(); i++)
    {
        start = std::max(dp[s[i]] + 1, start);
        dp[s[i]] = i;
        max = std::max(max, i - start + 1);
    }

    return max;
}

/********************/
/* 15. 3Sum */
vector<vector<int> > threeSum(vector<int> &num)
{
    std::sort(num.begin(), num.end());
    vector<vector<int>> result;
    
    for(int i = 0; i < num.size(); i++)
    {
        int target = -num[i];
        int left = i + 1;
        int right = num.size() - 1;
        while(left < right)
        {
            int sum = num[left] + num[right];
            if(sum > target)
            {
                right--;
            }
            else if(sum < target)
            {
                left++;
            }
            else
            {
                vector<int> temp;
                temp.push_back(num[i]);
                temp.push_back(num[left]);
                temp.push_back(num[right]);
                result.push_back(temp);

                while(left < right && num[left] == temp[1])
                {
                    left++;
                }
                while(left < right && num[right] == temp[2])
                {
                    right--;
                }
            }
        }

        while(i + 1 < num.size() && num[i+1] == num[i])
            i++;
    }

    return result;
}
/********************/
/* 20. Valid Parentheses */
bool isValid(string s)
{
    std::map<char, char> match_map;
    match_map['('] = ')';
    match_map['{'] = '}';
    match_map['['] = ']';

    stack<char> mystack; 
    for(char c : s)
    {
        if(match_map.find(c) != match_map.end())
        {
            mystack.push(c);
        }
        else
        {
            if(mystack.empty())
                return false;
            char matchChar = mystack.top();
            if(c != match_map[matchChar])
                return false;

            mystack.pop();
        }
    }

    return mystack.empty();
}
/********************/
/* 56. Merge Intervals */
struct Interval {
    int start;
    int end;
    Interval() : start(0), end(0) {}
    Interval(int s, int e) : start(s), end(e) {}
};

vector<Interval> merge(vector<Interval>& intervals) 
{
    std::sort(intervals.begin(), intervals.end(), [&](struct Interval a, struct Interval b)
    {
        return a.start < b.start;
    });

    for(auto it = intervals.begin(); it != intervals.end(); )
    {
        if((it + 1) != intervals.end() && (it + 1)->start <= it->end)
        {
            int nextEnd = (it + 1)->end;
            it->end = nextEnd > it->end ? nextEnd : it->end;
            intervals.erase(it + 1);
        }
        else
        {
            it++;
        }
    }

    return intervals;
}

/********************/
/* 21. Merge Two Sorted Lists */
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2)
{
    ListNode *head = new ListNode(0);
    ListNode *cur = head;
    ListNode *p1 = l1, *p2 = l2;

    while(p1 && p2)
    {
        cur->next = p1->val > p2->val ? p2 : p1;
        if(p1->val > p2->val)
        {
            cur->next = p2;
            p2 = p2->next;
        }
        else
        {
            cur->next = p1;
            p1 = p1->next;
        }
        
        cur = cur->next;
    }

    cur->next = p1 ? p1 : p2;

    return head->next; 
}

/********************/
/* 53. Maximum Subarray */
int maxSubArray(vector<int>& nums)
{
    int n = nums.size();
    int dp[n];
    for(int i = 0; i < n; i++)
        dp[i] = -INT64_MAX;

    dp[0] = nums[0];
    int max = nums[0];
    for(int i = 1; i < n; i++) 
    {
        dp[i] = dp[i - 1] > 0 ? dp[i - 1] + nums[i] : nums[i];
        if(dp[i] > max)
            max = dp[i];
    }

    return max;
}

/********************/
/* 253. Meeting Rooms II */
int minMeetingRooms(vector<Interval>& intervals)
{
    std::map<int, int> mymap;
    for(auto& i : intervals)
    {
        mymap[i.start]++;
        mymap[i.end]--;
    }

    int max = 0;
    int temp = 0;
    for(auto it = mymap.begin(); it != mymap.end(); it++)
    {
        temp += it->second; 
        if(temp > max)
            max = temp;
    }

    return max;
}

/********************/
/* 206. Reverse Linked List*/
ListNode* reverseList(ListNode* head)
{
    ListNode *prev = NULL;
    ListNode *next = NULL;

    while(head)
    {
        next = head->next;
        head->next = prev;
        prev = head;
        head = next; 
    }

    return prev; 
}

/********************/
/* 121. Best Time to Buy and Sell Stock*/
int maxProfit(vector<int>& prices)
{
    if(prices.size() == 0)
        return 0;
    int min = prices[0];
    int maxProfit = 0;
    for(int i = 1; i < prices.size(); i++)
    {
        if(prices[i] < min)
            min = prices[i];
        
        if(prices[i] - min > maxProfit)
            maxProfit = prices[i] - min;
    }
    return maxProfit;
}

/********************/
/* 7. Reverse Integer */
int reverse(int x)
{
    long long target = 0;
    while(x)
    {
        target = (x % 10) + target * 10;
        x /= 10;
    }

    return (target > INT32_MAX || target < INT32_MIN)? 0 : target; 
}
/********************/
/* 238. Product of Array Except Self*/
vector<int> productExceptSelf(vector<int>& nums) 
{
    int n = nums.size();
    int front[n], back[n];
    front[0] = nums[0];
    back[0] = nums[n - 1];

    for(int i = 1; i < n; i++)
    {
        front[i] = front[i - 1] * nums[i];
        back[i] = back[i - 1] * nums[n - i - 1];
    }

    vector<int> result;
    for(int i = 0; i < n; i++)
    {
        int allFront = i - 1 >= 0 ? front[i - 1] : 1;
        int allBack = i + 1 < n? back[n - 2 - i] : 1;
        result.push_back(allFront * allBack);
    }

    return result;
}

/********************/
/* 973. K Closest Points to Origin*/
vector<vector<int>> kClosest(vector<vector<int>>& points, int K)
{
    std::sort(points.begin(), points.end(), [&](vector<int>& a, vector<int>& b)
    {
        return (pow(a[0],2) + pow(a[1],2)) < (pow(b[0],2) + pow(b[1],2)); 
    });

    return vector<vector<int>>(points.begin(), points.begin() + K);
}

/********************/

int main(int argc, char **argv)
{
    /*vector<int> arr = {2, 7, 11, 15};
    int target = 9;
    vector<int> result;
    result = twoSum(arr, target);
    for(int i : result)
    {
        cout << " " << i << endl;
    }*/
    int a = -321;
    cout << "==>" << reverse(a);

    return 0;
}