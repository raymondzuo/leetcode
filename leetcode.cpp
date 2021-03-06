#include <iostream>
#include <stdio.h>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <set>
#include <stack>
#include <unordered_set>
#include <climits>
#include <unordered_map>
#include <stdio.h>
#include <sstream>
#include <deque>
#include <list>
#include <cmath>

using std::cin;
using std::cout;
using std::endl;
using std::map;
using std::max;
using std::pair;
using std::priority_queue;
using std::queue;
using std::set;
using std::stack;
using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::deque;
using std::list;

/**
 * leetcode most frequency problems
 * 
 */

//#1 Two sum
vector<int> twoSum(vector<int> &nums, int target)
{
    vector<int> result;
    std::map<int, int> mymap;

    for (int i = 0; i < nums.size(); i++)
    {
        int another = target - nums[i];
        auto it = mymap.find(another);
        if (it != mymap.end())
        {
            result.push_back(it->second);
            result.push_back(i);
            return result;
        }

        mymap[nums[i]] = i;
    }
}

struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
//#2 Add two numbers
ListNode *addTwoNumbers(ListNode *l1, ListNode *l2)
{
    ListNode *head = new ListNode(0);
    ListNode *cur = head;
    int carry = 0;
    ListNode *p1 = l1, *p2 = l2;

    while (p1 || p2)
    {
        int x = p1 ? p1->val : 0;
        int y = p2 ? p2->val : 0;

        int sum = x + y + carry;
        carry = sum / 10;
        sum = sum % 10;
        ListNode *temp = new ListNode(sum);

        cur->next = temp;
        cur = temp;

        if (p1)
            p1 = p1->next;
        if (p2)
            p2 = p2->next;
    }

    if (carry > 0)
        cur->next = new ListNode(carry);

    return head->next;
}

/********************/
//#200. Number of Islands
void dfs(vector<vector<char>> &grid, int r, int c)
{
    int maxr = grid.size();
    int maxc = grid[0].size();

    if (r < 0 || c < 0 || r >= grid.size() || c >= grid[0].size() || grid[r][c] == '0')
        return;
    grid[r][c] = '0';
    dfs(grid, r + 1, c);
    dfs(grid, r - 1, c);
    dfs(grid, r, c + 1);
    dfs(grid, r, c - 1);
}

int numIslands(vector<vector<char>> &grid)
{
    int count = 0;

    int maxr = grid.size();
    if (0 == maxr)
        return count;
    int maxc = grid[0].size();
    if (0 == maxc)
        return count;

    for (int i = 0; i < maxr; i++)
    {
        for (int j = 0; j < maxc; j++)
        {
            if ('1' == grid[i][j])
            {
                count++;
                dfs(grid, i, j);
            }
        }
    }

    return count;
}

int numIslandsBFS(vector<vector<char>> &grid)
{
    int nr = grid.size();
    if (0 == nr)
        return 0;
    int nc = grid[0].size();

    int count = 0;
    for (int r = 0; r < nr; r++)
    {
        for (int c = 0; c < nc; c++)
        {
            if ('1' == grid[r][c])
            {
                count++;
                grid[r][c] = '0';
                queue<pair<int, int>> tempqueue;
                tempqueue.push({r, c});

                while (!tempqueue.empty())
                {
                    auto front = tempqueue.front();
                    tempqueue.pop();
                    int tempr = front.first;
                    int tempc = front.second;

                    if (tempr - 1 >= 0 && grid[tempr - 1][tempc] == '1')
                    {
                        tempqueue.push({tempr - 1, tempc});
                        grid[tempr - 1][tempc] = '0';
                    }
                    if (tempr + 1 < nr && grid[tempr + 1][tempc] == '1')
                    {
                        tempqueue.push({tempr + 1, tempc});
                        grid[tempr + 1][tempc] = '0';
                    }
                    if (tempc - 1 >= 0 && grid[tempr][tempc - 1] == '1')
                    {
                        tempqueue.push({tempr, tempc - 1});
                        grid[tempr][tempc - 1] = '0';
                    }
                    if (tempc + 1 < nc && grid[tempr][tempc + 1] == '1')
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
    int si = 0;   //start
    int dp[n][n]; //n->n from n to n
    //length is 1
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                dp[i][j] = 1;
            else
                dp[i][j] = 0;
        }
    }
    //length is 2
    for (int i = 0; i < n - 1; i++)
    {
        if (s[i + 1] == s[i])
        {
            dp[i][i + 1] = 1;
            max = 2;
            si = i;
        }
    }

    //length large than 3
    for (int k = 3; k <= n; k++)
    {
        for (int i = 0; i < n - k + 1; i++)
        {
            int j = i + k - 1;
            if (dp[i + 1][j - 1] && s[i] == s[j])
            {
                dp[i][j] = 1;
                if (k > max)
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

    for (int i = 0; i < s.length(); i++)
    {
        start = std::max(dp[s[i]] + 1, start);
        dp[s[i]] = i;
        max = std::max(max, i - start + 1);
    }

    return max;
}

/********************/
/* 15. 3Sum */
vector<vector<int>> threeSum(vector<int> &num)
{
    std::sort(num.begin(), num.end());
    vector<vector<int>> result;

    for (int i = 0; i < num.size(); i++)
    {
        int target = -num[i];
        int left = i + 1;
        int right = num.size() - 1;
        while (left < right)
        {
            int sum = num[left] + num[right];
            if (sum > target)
            {
                right--;
            }
            else if (sum < target)
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

                while (left < right && num[left] == temp[1])
                {
                    left++;
                }
                while (left < right && num[right] == temp[2])
                {
                    right--;
                }
            }
        }

        while (i + 1 < num.size() && num[i + 1] == num[i])
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
    for (char c : s)
    {
        if (match_map.find(c) != match_map.end())
        {
            mystack.push(c);
        }
        else
        {
            if (mystack.empty())
                return false;
            char matchChar = mystack.top();
            if (c != match_map[matchChar])
                return false;

            mystack.pop();
        }
    }

    return mystack.empty();
}
/********************/
/* 56. Merge Intervals */
struct Interval
{
    int start;
    int end;
    Interval() : start(0), end(0) {}
    Interval(int s, int e) : start(s), end(e) {}
};

vector<Interval> merge(vector<Interval> &intervals)
{
    std::sort(intervals.begin(), intervals.end(), [&](struct Interval a, struct Interval b) {
        return a.start < b.start;
    });

    for (auto it = intervals.begin(); it != intervals.end();)
    {
        if ((it + 1) != intervals.end() && (it + 1)->start <= it->end)
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
ListNode *mergeTwoLists(ListNode *l1, ListNode *l2)
{
    ListNode *head = new ListNode(0);
    ListNode *cur = head;
    ListNode *p1 = l1, *p2 = l2;

    while (p1 && p2)
    {
        cur->next = p1->val > p2->val ? p2 : p1;
        if (p1->val > p2->val)
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
int maxSubArray(vector<int> &nums)
{
    int n = nums.size();
    int dp[n];
    for (int i = 0; i < n; i++)
        dp[i] = -INT64_MAX;

    dp[0] = nums[0];
    int max = nums[0];
    for (int i = 1; i < n; i++)
    {
        dp[i] = dp[i - 1] > 0 ? dp[i - 1] + nums[i] : nums[i];
        if (dp[i] > max)
            max = dp[i];
    }

    return max;
}

/********************/
/* 253. Meeting Rooms II */
int minMeetingRooms(vector<Interval> &intervals)
{
    std::map<int, int> mymap;
    for (auto &i : intervals)
    {
        mymap[i.start]++;
        mymap[i.end]--;
    }

    int max = 0;
    int temp = 0;
    for (auto it = mymap.begin(); it != mymap.end(); it++)
    {
        temp += it->second;
        if (temp > max)
            max = temp;
    }

    return max;
}

/********************/
/* 206. Reverse Linked List*/
ListNode *reverseList(ListNode *head)
{
    ListNode *prev = NULL;
    ListNode *next = NULL;

    while (head)
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
int maxProfit(vector<int> &prices)
{
    if (prices.size() == 0)
        return 0;
    int min = prices[0];
    int maxProfit = 0;
    for (int i = 1; i < prices.size(); i++)
    {
        if (prices[i] < min)
            min = prices[i];

        if (prices[i] - min > maxProfit)
            maxProfit = prices[i] - min;
    }
    return maxProfit;
}

/********************/
/* 7. Reverse Integer */
int reverse(int x)
{
    long long target = 0;
    while (x)
    {
        target = (x % 10) + target * 10;
        x /= 10;
    }

    return (target > INT32_MAX || target < INT32_MIN) ? 0 : target;
}
/********************/
/* 238. Product of Array Except Self*/
vector<int> productExceptSelf(vector<int> &nums)
{
    int n = nums.size();
    int front[n], back[n];
    front[0] = nums[0];
    back[0] = nums[n - 1];

    for (int i = 1; i < n; i++)
    {
        front[i] = front[i - 1] * nums[i];
        back[i] = back[i - 1] * nums[n - i - 1];
    }

    vector<int> result;
    for (int i = 0; i < n; i++)
    {
        int allFront = i - 1 >= 0 ? front[i - 1] : 1;
        int allBack = i + 1 < n ? back[n - 2 - i] : 1;
        result.push_back(allFront * allBack);
    }

    return result;
}

/********************/
/* 973. K Closest Points to Origin*/
vector<vector<int>> kClosest(vector<vector<int>> &points, int K)
{
    std::sort(points.begin(), points.end(), [&](vector<int> &a, vector<int> &b) {
        return (pow(a[0], 2) + pow(a[1], 2)) < (pow(b[0], 2) + pow(b[1], 2));
    });

    return vector<vector<int>>(points.begin(), points.begin() + K);
}

/********************/
/* 11. Container With Most Water*/
int maxArea(vector<int> &height)
{
    int s = 0, e = height.size() - 1;
    int max = 0;

    while (s < e)
    {
        int len = e - s;
        int min = 0;
        if (height[s] < height[e])
        {
            min = height[s];
            s++;
        }
        else
        {
            min = height[e];
            e--;
        }

        int temp_area = len * min;
        if (temp_area > max)
            max = temp_area;
    }

    return max;
}

/********************/
/* 33. Search in Rotated Sorted Array */
int search_rotate_index(vector<int> &arr, int low, int high)
{
    if (arr[low] < arr[high])
        return 0;

    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (arr[mid] > arr[mid + 1])
        {
            return mid + 1;
        }
        else
        {
            if (arr[mid] >= arr[low])
                low = mid + 1;
            else
            {
                high = mid - 1;
            }
        }
    }
}

int binary_search(vector<int> &arr, int low, int high, int target)
{
    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (arr[mid] == target)
            return mid;
        else
        {
            if (arr[mid] < target)
            {
                low = mid + 1;
            }
            else
            {
                high = mid - 1;
            }
        }
    }

    return -1;
}

int search(vector<int> &arr, int target)
{
    int n = arr.size();
    if (n == 0)
        return -1;
    if (n == 1)
        return arr[0] == target ? 0 : -1;
    int rotate_index = search_rotate_index(arr, 0, n - 1);
    if (arr[rotate_index] == target)
        return rotate_index;
    if (rotate_index == 0)
        return binary_search(arr, 0, n - 1, target);

    if (target < arr[0])
        return binary_search(arr, rotate_index, n - 1, target);
    else
        return binary_search(arr, 0, rotate_index - 1, target);
}

/********************/
/* 102. Binary Tree Level Order Traversal*/
struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

vector<vector<int>> levelOrder(TreeNode *root)
{
    vector<vector<int>> result;
    if (NULL == root)
        return result;
    queue<TreeNode *> myqueue;
    myqueue.push(root);

    while (!myqueue.empty())
    {
        vector<int> level_vec;
        int len = myqueue.size();
        for (int i = 0; i < len; i++)
        {
            TreeNode *front = myqueue.front();
            level_vec.push_back(front->val);
            myqueue.pop();

            if (NULL != front->left)
                myqueue.push(front->left);
            if (NULL != front->right)
                myqueue.push(front->right);
        }

        result.push_back(level_vec);
    }

    return result;
}

void levelOrderHelper(vector<vector<int>> &result, vector<TreeNode> &nodes)
{
    if (nodes.size() == 0)
        return;

    vector<int> levelNodes;
    vector<TreeNode> nexLevelNodes;

    for (TreeNode node : nodes)
    {
        levelNodes.push_back(node.val);
        if (node.left != NULL)
            nexLevelNodes.push_back(*node.left);
        if (node.right != NULL)
            nexLevelNodes.push_back(*node.right);
    }
    result.push_back(levelNodes);

    levelOrderHelper(result, nexLevelNodes);
}

vector<vector<int>> levelOrderRecursion(TreeNode *root)
{
    vector<vector<int>> result;
    if (NULL == root)
        return result;

    vector<TreeNode> init;
    init.push_back(*root);

    levelOrderHelper(result, init);

    return result;
}
/********************/
/* 22. Generate Parentheses */
void back_track(vector<string> &result, string &strTemp, int open, int close, int max)
{
    if (strTemp.length() == 2 * max)
    {
        result.push_back(strTemp);
        strTemp.clear();
        return;
    }

    if (open < max)
        back_track(result, strTemp.append("("), open + 1, close, max);
    if (close < open)
        back_track(result, strTemp.append(")"), open, close + 1, max);
}

vector<string> generateParenthesis(int n)
{
    vector<string> result;
    string strTemp;
    back_track(result, strTemp, 0, 0, n);

    return result;
}

/********************/
/* 771. Jewels and Stones*/
int numJewelsInStones(string J, string S)
{
    int count = 0;
    unordered_set<char> myset;
    for (char c : J)
        myset.insert(c);
    for (char c : S)
    {
        if (myset.find(c) != myset.end())
        {
            count++;
        }
    }

    return count;
}

/********************/
/* 141. Linked List Cycle*/
struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

bool hasCycle(ListNode *head)
{
    if (head == NULL || head->next == NULL)
        return false;

    ListNode *slow = head, *fast = head->next;
    while (slow != NULL && fast != NULL)
    {
        if (slow == fast)
            return true;
        slow = slow->next;
        fast = fast->next;
        if (fast != NULL)
            fast = fast->next;
    }

    return false;
}

/********************/
/* 9. Palindrome Number */
bool isPalindrome(int x)
{
    if (x < 0)
        return false;

    long long temp = 0;
    int tempx = x;
    while (x != 0)
    {
        temp = x % 10 + temp * 10;
        x /= 10;
    }
    if (temp > INT_MAX)
        temp = 0;
    return tempx == temp;
}

/********************/
/* 322. Coin Change*/
int coinChange(vector<int> &coins, int amount)
{
    int dp[amount + 1];
    for (int i = 0; i <= amount; i++)
        dp[i] = amount + 1;

    dp[0] = 0;
    for (int i = 1; i <= amount; i++)
    {
        for (int j = 0; j < coins.size(); j++)
        {
            if (i >= coins[j])
                dp[i] = std::min(dp[i], dp[i - coins[j]] + 1);
        }
    }

    return dp[amount] > amount ? -1 : dp[amount];
}

/********************/
/* 560. Subarray Sum Equals K */
int subarraySum(vector<int> &nums, int k)
{
    int count = 0;
    int n = nums.size();
    int sum[n + 1];
    sum[0] = 0;
    for (int i = 1; i <= nums.size(); i++)
        sum[i] = sum[i - 1] + nums[i - 1];

    for (int i = 0; i <= n; i++)
    {
        for (int j = i + 1; j <= n; j++)
        {
            if (sum[j] - sum[i] == k)
                count++;
        }
    }

    return count;
}

int subarraySumHash(vector<int> &nums, int k)
{
    unordered_map<int, int> sum_map;
    int sum = 0;
    int count = 0;
    sum_map[0] = 1;

    for (int i = 1; i <= nums.size(); i++)
    {
        sum += nums[i - 1];
        count += sum_map[sum - k];
        ++sum_map[sum];
    }

    return count;
}
/********************/
/* 560. Subarray Sum Equals K */
vector<int> spiralOrder(vector<vector<int>> &matrix)
{
    vector<int> res;
    if (matrix.size() == 0)
        return res;

    int r1 = 0, r2 = matrix.size() - 1;
    int c1 = 0, c2 = matrix[0].size() - 1;

    while (r1 < r2 && c1 < c2)
    {
        for (int i = c1; i < c2; i++)
            res.push_back(matrix[r1][i]);
        for (int i = r1; i < r2; i++)
            res.push_back(matrix[i][c2]);
        for (int i = c2; i > c1; i--)
            res.push_back(matrix[r2][i]);
        for (int i = r2; i > r1; i--)
            res.push_back(matrix[i][c1]);

        c1++;
        c2--;
        r1++;
        r2--;
    }

    return res;
}

/********************/
/* 412. Fizz Buzz */
vector<string> fizzBuzz(int n)
{
    vector<string> res;
    bool mul3 = false, mul5 = false;

    for (int i = 1; i <= n; i++)
    {

        mul3 = false;
        mul5 = false;
        if (i % 3 == 0)
            mul3 = true;
        if (i % 5 == 0)
            mul5 = true;

        if (mul3 && mul5)
            res.push_back("FizzBuzz");
        else if (mul3)
            res.push_back("Fizz");
        else if (mul5)
            res.push_back("Buzz");
        else
            res.push_back(std::to_string(i));
    }

    return res;
}

/********************/
/* 46. Permutations */

void permute_recurse(vector<int> &nums, int begin, vector<vector<int>> &result)
{
    if (begin >= nums.size())
    {
        result.push_back(nums);
        return;
    }

    for (int i = begin; i < nums.size(); i++)
    {
        std::swap(nums[begin], nums[i]);
        permute_recurse(nums, begin + 1, result);
        //reset
        std::swap(nums[begin], nums[i]);
    }
}

vector<vector<int>> permute(vector<int> &nums)
{
    vector<vector<int>> result;
    permute_recurse(nums, 0, result);
    return result;
}

/********************/
/* 88. Merge Sorted Array */
void merge(vector<int> &nums1, int m, vector<int> &nums2, int n)
{
    vector<int> temp;
    for (int i = 0; i < m; i++)
        temp.push_back(nums1[i]);
    int i = 0, j = 0, k = 0;
    while (i < temp.size() && j < nums2.size())
    {
        if (temp[i] <= nums2[j])
        {
            nums1[k] = temp[i];
            i++;
        }
        else if (temp[i] > nums2[j])
        {
            nums1[k] = nums2[j];
            j++;
        }

        k++;
    }

    if (i < temp.size())
    {
        nums1[k] = temp[i];
        i++;
        k++;
    }
    if (j < nums2.size())
    {
        nums1[k] = nums2[i];
        j++;
        k++;
    }
}

/********************/
/* 160. Intersection of Two Linked Lists*/
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB)
{
    ListNode *pa = headA;
    ListNode *pb = headB;

    if (pa == NULL || pb == NULL)
        return NULL;

    while (pa != NULL && pb != NULL && pa != pb)
    {
        pa = pa->next;
        pb = pb->next;

        if (pa == pb)
            return pa;

        if (pa == NULL)
            pa = headB;
        if (pb == NULL)
            pb = headA;
    }

    return pa;
}
/********************/
/* 234. Palindrome Linked List*/
ListNode *temp;
bool isPalindrome(ListNode *head)
{
    temp = head;
    return check(head);
}

bool check(ListNode *node)
{
    if (node == NULL)
        return true;
    bool isPal = check(node->next) && (temp->val == node->val);
    temp = temp->next;
    return isPal;
}

/********************/
/* 23. Merge k Sorted Lists*/
ListNode *merge2Lists(ListNode *l1, ListNode *l2)
{
    if (l1 == NULL)
        return l2;
    if (l2 == NULL)
        return l1;

    if (l1->val <= l2->val)
    {
        l1->next = merge2Lists(l1->next, l2);
        return l1;
    }
    else
    {
        l2->next = merge2Lists(l1, l2->next);
        return l2;
    }
}

ListNode *mergeKLists(vector<ListNode *> &lists)
{
    if (lists.empty())
        return NULL;
    //straight solution
    while (lists.size() > 1)
    {
        lists.push_back(merge2Lists(lists[0], lists[1]));
        lists.erase(lists.begin());
        lists.erase(lists.begin());
    }

    return lists.front();
}

struct compare
{
    bool operator()(ListNode *n1, ListNode *n2)
    {
        return n1->val > n2->val;
    }
};

ListNode *mergeKListsPQ(vector<ListNode *> &lists)
{
    priority_queue<ListNode *, vector<ListNode *>, compare> q;
    for (auto l : lists)
    {
        if (l)
            q.push(l);
    }

    if (q.empty())
        return NULL;

    ListNode *result = q.top();
    ListNode *tail = result;
    q.pop();
    if (result->next)
        q.push(result->next);
    while (!q.empty())
    {
        tail->next = q.top();
        q.pop();
        tail = tail->next;
        if (tail->next)
            q.push(tail->next);
    }

    return result;
}

/********************/
/* 344. Reverse String*/
void reverseString(vector<char> &s)
{
    int i = 0, j = s.size() - 1;
    while (i < j)
    {
        s[i] = s[i] + s[j];
        s[j] = s[i] - s[j];
        s[j] = s[i] - s[j];

        i++;
        j--;
    }
}

/********************/
/* 79. Word Search*/
bool dfs_find(vector<vector<char>> &board, const string &word, int index, int i, int j)
{
    if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size() || index >= word.size() || board[i][j] == '\0')
        return false;

    if (word[index] != board[i][j])
        return false;
    if (index == word.size() - 1)
        return true;

    char temp = board[i][j];
    board[i][j] = '\0';

    if (dfs_find(board, word, index + 1, i + 1, j) || dfs_find(board, word, index + 1, i - 1, j) || dfs_find(board, word, index + 1, i, j + 1) || dfs_find(board, word, index + 1, i, j - 1))
        return true;

    board[i][j] = temp;
    return false;
}

bool exist(vector<vector<char>> &board, string word)
{
    if (word.empty())
        return true;
    if (board.size() == 0)
        return false;

    int r = board.size();
    int c = board[0].size();

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            bool res = dfs_find(board, word, 0, i, j);
            if (res)
                return res;
        }
    }

    return false;
}

/********************/
/* 203. Remove Linked List Elements*/
ListNode *removeElements(ListNode *head, int val)
{
    ListNode **ppl = &head;

    while (*ppl != NULL)
    {
        if ((*ppl)->val == val)
        {
            *ppl = (*ppl)->next;
        }
        else
        {
            ppl = &(*ppl)->next;
        }
    }

    return head;
}
/********************/
/* 103. Binary Tree Zigzag Level Order Traversal*/
vector<vector<int>> zigzagLevelOrder(TreeNode *root)
{
    vector<vector<int>> result;
    if (NULL == root)
        return result;

    stack<TreeNode *> st1;
    stack<TreeNode *> st2;
    int lvl = 0;

    st1.push(root);
    while (1)
    {
        if (st1.empty() && st2.empty())
            break;

        vector<int> tempVec;

        while (!st1.empty())
        {
            TreeNode *pNode = st1.top();
            tempVec.push_back(pNode->val);
            if (pNode->left)
                st2.push(pNode->left);
            if (pNode->right)
                st2.push(pNode->right);
            st1.pop();
        }

        if (tempVec.size() > 0)
        {
            result.push_back(tempVec);
            tempVec.clear();
        }

        while (!st2.empty())
        {
            TreeNode *pNode = st2.top();
            tempVec.push_back(pNode->val);
            if (pNode->left)
                st1.push(pNode->left);
            if (pNode->right)
                st1.push(pNode->right);
            st2.pop();
        }

        if (tempVec.size() > 0)
            result.push_back(tempVec);
    }

    return result;
}
/********************/
/* 215. Kth Largest Element in an Array*/
int findKthLargest(vector<int> &nums, int k)
{
    if (0 == nums.size())
        return 0;

    priority_queue<int, vector<int>, std::greater<int>> pq;
    for (int i = 0; i < nums.size(); i++)
    {
        pq.push(nums[i]);
        if (pq.size() > k)
            pq.pop();
    }

    return pq.top();
}
/********************/
/* 138. Copy List with Random Pointer*/
class Node
{
  public:
    int val;
    Node *next;
    Node *random;

    Node() {}

    Node(int _val, Node *_next, Node *_random)
    {
        val = _val;
        next = _next;
        random = _random;
    }
};

map<Node *, Node *> clone_node_map;
Node *get_clone_node(Node *oldNode)
{
    if (NULL == oldNode)
        return NULL;
    if (clone_node_map.find(oldNode) != clone_node_map.end())
    {
        return clone_node_map[oldNode];
    }
    else
    {
        clone_node_map[oldNode] = new Node(oldNode->val, NULL, NULL);
        return clone_node_map[oldNode];
    }
}

Node *copyRandomList(Node *head)
{
    if (NULL == head)
        return head;

    Node *pNewHead = new Node(head->val, NULL, NULL);
    clone_node_map[head] = pNewHead;
    Node *pOldHead = head;

    while (pOldHead)
    {
        pNewHead->next = get_clone_node(pOldHead->next);
        pNewHead->random = get_clone_node(pOldHead->random);

        pOldHead = pOldHead->next;
        pNewHead = pNewHead->next;
    }

    return clone_node_map[head];
}
/********************/
/* 98. Validate Binary Search Tree*/
void get_inorder(TreeNode *root, vector<TreeNode *> &vecInorder)
{
    if (root == NULL)
        return;

    get_inorder(root->left, vecInorder);
    vecInorder.push_back(root);
    get_inorder(root->right, vecInorder);
}

bool isValidBST(TreeNode *root)
{
    if (NULL == root)
        return true;

    vector<TreeNode *> vecInorder;
    get_inorder(root, vecInorder);

    for (int i = 0; i < vecInorder.size() - 1; i++)
    {
        if (vecInorder[i]->val >= vecInorder[i + 1]->val)
            return false;
    }

    return true;
}
/********************/
/* 226. Invert Binary Tree*/
TreeNode *invertTree(TreeNode *root)
{
    if (root == NULL)
        return NULL;
    TreeNode *pTemp = root->left;
    root->left = invertTree(root->right);
    root->right = invertTree(pTemp);

    return root;
}

TreeNode *invertTreeIterative(TreeNode *root){
    if(NULL == root)
        return root;
    queue<TreeNode *> q;
    q.push(root);

    while(!q.empty()){
        TreeNode *top = q.front();
        TreeNode *temp = top->left;
        top->left = top->right;
        top->right = temp;
        q.pop();

        if(top->left)
            q.push(top->left);
        if(top->right)
            q.push(top->right);
    }

    return root;
}

/********************/
/* 543. Diameter of Binary Tree*/
int calTreeNodeLen(TreeNode *root, int &max)
{
    if (NULL == root)
        return 0;

    int left_len = calTreeNodeLen(root->left, max);
    int right_len = calTreeNodeLen(root->right, max);

    max = std::max(max, left_len + right_len + 1);
    return std::max(left_len, right_len) + 1;
}

int diameterOfBinaryTree(TreeNode *root)
{
    if (NULL == root)
        return 0;

    int max = 1;
    calTreeNodeLen(root, max);

    return max - 1;
}
/********************/
/* 199. Binary Tree Right Side View*/
vector<int> rightSideView(TreeNode *root)
{
    vector<int> res;
    if (NULL == root)
        return res;

    queue<TreeNode *> node_queue;
    node_queue.push(root);

    while (!node_queue.empty())
    {
        int len = node_queue.size();
        for (int i = 0; i < len; i++)
        {
            TreeNode *pTemp = node_queue.front();
            if (pTemp->left)
                node_queue.push(pTemp->left);
            if (pTemp->right)
                node_queue.push(pTemp->right);
            if (i == len - 1)
                res.push_back(pTemp->val);

            node_queue.pop();
        }
    }

    return res;
}
/********************/
/* 94. Binary Tree Inorder Traversal*/
vector<int> inorderTraversal(TreeNode *root)
{
    vector<int> res;
    stack<TreeNode *> st;
    TreeNode *pTempRoot = root;

    while (pTempRoot || !st.empty())
    {
        while (pTempRoot)
        {
            st.push(pTempRoot);
            pTempRoot = pTempRoot->left;
        }

        pTempRoot = st.top();
        res.push_back(pTempRoot->val);
        st.pop();

        pTempRoot = pTempRoot->right;
    }

    return res;
}

/********************/
/* 124. Binary Tree Maximum Path Sum*/
int getMaxSum(TreeNode *root, int &max)
{
    if (root == NULL)
        return 0;

    int left_max = getMaxSum(root->left, max);
    left_max = left_max > 0 ? left_max : 0;
    int right_max = getMaxSum(root->right, max);
    right_max = right_max > 0 ? right_max : 0;

    int temp = root->val + left_max + right_max;
    if (temp > max)
        max = temp;

    return root->val + std::max(left_max, right_max);
}

int maxPathSum(TreeNode *root)
{
    int max = INT_MIN;
    getMaxSum(root, max);
    return max;
}
/********************/
/* 236. Lowest Common Ancestor of a Binary Tree*/
bool find_path(TreeNode *root, TreeNode *pTargetNode, vector<TreeNode *> &vecPath)
{
    if (NULL == root)
        return false;

    vecPath.push_back(root);
    if (root == pTargetNode)
        return true;
    else
    {
        bool bFindLeft = find_path(root->left, pTargetNode, vecPath);
        if (!bFindLeft)
        {
            bool bFindRight = find_path(root->right, pTargetNode, vecPath);
            if (!bFindRight)
            {
                vecPath.pop_back();
                return false;
            }
        }
    }

    return true;
}

TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q)
{
    if (NULL == root)
        return NULL;
    vector<TreeNode *> vec_p, vec_q;
    bool bLeft = find_path(root, p, vec_p);
    bool bRight = find_path(root, q, vec_q);
    if (bLeft && bRight)
    {
        int i = 0, j = 0;
        TreeNode *lca = root;
        while (i < vec_p.size() && j < vec_q.size())
        {
            if (vec_p[i] == vec_q[j])
            {
                lca = vec_p[i];
                i++;
                j++;
            }
            else
                break;
        }

        return lca;
    }

    return NULL;
}
/********************/
/* 295. Find Median from Data Stream*/
class MedianFinder
{
  public:
    /** initialize your data structure here. */
    MedianFinder()
    {
    }

    void addNum(int num)
    {
        data.insert(std::lower_bound(data.begin(), data.end(), num), num);
    }

    double findMedian()
    {
        int n = data.size();
        if (n & 1)
            return data[n / 2];
        else
        {
            return ((double)data[n / 2] + (double)data[n / 2 - 1]) / 2.0;
        }
    }

  private:
    vector<int> data;
};
class MedianFinder2Heaps //with two heap
{
  public:
    /** initialize your data structure here. */
    MedianFinder2Heaps()
    {
    }

    void addNum(int num)
    {
        low.push(num);

        high.push(low.top());
        low.pop();

        if (low.size() < high.size())
        {
            low.push(high.top());
            high.pop();
        }
    }

    double findMedian()
    {
        return low.size() > high.size() ? low.top() : (low.top() + high.top()) * 0.5;
    }

  private:
    priority_queue<int> low;
    priority_queue<int, vector<int>, std::greater<int>> high;
};
/********************/
/* 301. Remove Invalid Parentheses */
vector<string> removeInvalidParentheses(string s)
{
    map<char, int> par_map;
    par_map['('] = -1;
    par_map[')'] = 1;

    int sum = 0;
    for (char c : s)
    {
        if (par_map.find(c) != par_map.end())
        {
            sum += par_map[c];
            if (c == ')')
            {
                //if(sum > 0)//delete ')'
            }
        }
    }
}

/********************/
/* 17. Letter Combinations of a Phone Number*/
map<char, string> num_map;
void constructStr(const string &digits, int index, string strRes, vector<string> &vecRes)
{
    if (strRes.size() == digits.size())
    {
        vecRes.push_back(strRes);
        return;
    }

    char cur_ch = digits[index];
    const string &cur_str = num_map[cur_ch];

    for (char c : cur_str)
    {
        constructStr(digits, index + 1, strRes + c, vecRes);
    }
}

vector<string> letterCombinations(string digits)
{
    vector<string> res;
    if (digits.empty())
        return res;

    num_map['2'] = "abc";
    num_map['3'] = "def";
    num_map['4'] = "ghi";
    num_map['5'] = "jkl";
    num_map['6'] = "mno";
    num_map['7'] = "pqrs";
    num_map['8'] = "tuv";
    num_map['9'] = "wxyz";

    constructStr(digits, 0, "", res);

    return res;
}
/********************/
/* 380. Insert Delete GetRandom O(1)*/
class RandomizedSet
{
  public:
    /** Initialize your data structure here. */
    RandomizedSet()
    {
    }

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    bool insert(int val)
    {
        if (idx_map.find(val) != idx_map.end())
            return false;
        nums.emplace_back(val);
        idx_map[val] = nums.size() - 1;
        return true;
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    bool remove(int val)
    {
        if (idx_map.find(val) == idx_map.end())
            return false;
        int last = nums.back();
        idx_map[last] = idx_map[val];
        nums[idx_map[last]] = last;
        nums.pop_back();
        idx_map.erase(val);
        return true;
    }

    /** Get a random element from the set. */
    int getRandom()
    {
        return nums[rand() % nums.size()];
    }

  private:
    vector<int> nums;
    unordered_map<int, int> idx_map;
};

/********************/
/* 139. Word Break*/

/********************/
/* 946. Validate Stack Sequences*/
bool validateStackSequences(vector<int>& pushed, vector<int>& popped)
{
    if(pushed.size() == 0 && popped.size() == 0)
        return true;

    if(pushed.size() == 0 || popped.size() == 0)
        return false;

    bool bValid = true;
    
    stack<int> st_data;
    int j = 0;
    for(int i = 0; i < popped.size(); i++)
    {
        while(st_data.size() == 0 || st_data.top() != popped[i]) 
        {
            if(j == pushed.size())
                break;

            st_data.push(pushed[j]);            
            j++;
        }

        if(st_data.top() != popped[i])
        {
            bValid = false;
            break;
        }
        
        st_data.pop();
    }

    return bValid; 
}

/********************/
/* 255. Verify Preorder Sequence in Binary Search Tree*/
bool verifyPreorderHelper(vector<int>& preorder, int s, int e)
{
    if(preorder.size() == 0)
        return true;
    if(s >= e)
        return true;

    int nRoot = preorder[s]; //root value
    int left = s + 1; 
    for(;left < e; left++)
    {
        if(preorder[left] > nRoot)
            break;
    }

    for(int j = left; j < e; j++)
    {
        if(preorder[j] < nRoot)
            return false;
    }

    bool bLeftValid = verifyPreorderHelper(preorder, s + 1, left);
    if(bLeftValid)
    {
        bool bRightValid = verifyPreorderHelper(preorder, left, e);
        return bRightValid;
    }

    return false;
}

bool verifyPreorder(vector<int>& preorder)
{
    return verifyPreorderHelper(preorder, 0, preorder.size());
}

/********************/
/* 113. Path Sum II*/
void pathSumHelper(TreeNode *root, int sum, int curSum, vector<int>& vecTemp, vector<vector<int> >& res)
{
    if(NULL == root)
        return;

    vecTemp.push_back(root->val);
    curSum += root->val;
    bool bLeaf = !root->left && !root->right;
    if(bLeaf && curSum == sum)
    {
        res.push_back(vecTemp); 
    }
    else
    {
        pathSumHelper(root->left, sum ,curSum, vecTemp, res);
        pathSumHelper(root->right, sum ,curSum, vecTemp, res);
    }

    vecTemp.pop_back();
}

vector<vector<int>> pathSum(TreeNode* root, int sum) 
{
    vector<vector<int> > res;
    if(NULL == root)
        return res;
    vector<int> temp;
    int curSum = 0;
    pathSumHelper(root, sum, curSum, temp, res);
    return res;
}

/********************/
/* 426. Convert Binary Search Tree to Sorted Doubly Linked List*/
void treeToDoublyListHelper(TreeNode *root, TreeNode **ppLastNode)
{
    if(NULL == root)
        return;

    if(root->left)
        treeToDoublyListHelper(root->left, ppLastNode);

    root->left = *ppLastNode;
    if(*ppLastNode)
        (*ppLastNode)->right = root;

    *ppLastNode = root;

    if(root->right)
        treeToDoublyListHelper(root->right, ppLastNode);
}

TreeNode* treeToDoublyList(TreeNode* root) 
{
    if(NULL == root)
        return NULL;
    TreeNode *ppLastNode = NULL;
    treeToDoublyListHelper(root, &ppLastNode);

    TreeNode *pCpLastNode = ppLastNode;
    while(1)
    {
        TreeNode *pLeft = (ppLastNode)->left;
        if(pLeft)
            (ppLastNode) = pLeft;
        else
        {
            ppLastNode->left = pCpLastNode;
            pCpLastNode->right = ppLastNode;
            return ppLastNode;
        }
    }

    return NULL;
}
/********************/
/* 136. Single Number*/
int singleNumber(vector<int>& nums) 
{
    if(nums.size() == 0)
        return 0;

    int res = nums[0];
    for(int i = 1; i < nums.size(); i++)
    {
        res ^= nums[i];
    }

    return res;
}
/********************/
/* 264. Ugly Number II*/
int nthUglyNumber(int n)
{
    if(0 >= n)
        return 0;
    if(1 == n)
        return 1;
    vector<int> vecUgly(n);

    int p2 = 0, p3 = 0, p5 = 0;
    vecUgly[0] = 1;

    for(int i = 1; i < n; i++)
    {
        vecUgly[i] = std::min(vecUgly[p2] * 2, std::min(vecUgly[p3] * 3, vecUgly[p5] * 5));
        if(vecUgly[i] == vecUgly[p2] * 2)
            p2++;
        if(vecUgly[i] == vecUgly[p3] * 3)
            p3++;
        if(vecUgly[i] == vecUgly[p5] * 5)
            p5++;
    }

    return vecUgly[n - 1];
}
/********************/
/* 179. Largest Number*/
string largestNumber(vector<int>& nums)
{
    string result = "0";
    if(0 == nums.size())
        return result;

    std::sort(nums.begin(), nums.end(), [&](int a, int b)
    {
        string tempAb = std::to_string(a) + std::to_string(b);
        string tempBa = std::to_string(b) + std::to_string(a);
        return tempAb > tempBa;
    });

    for(int i = 0; i < nums.size(); i++)
    {
        if(result == "0")
            result = "";
        result += std::to_string(nums[i]);
    }
    return result;
}
/********************/
/* 387. First Unique Character in a String*/
int firstUniqChar(string s)
{
    int ch_map[256];
    for(int i = 0; i < 256; i++)
        ch_map[i] = 0;
    for(char c : s)
    {
        ch_map[c]++;
    }

    for(char c : s)
        if(ch_map[c] == 1)
            return s.find_first_of(c, 0);

    return -1;
}

/********************/
/* 49. Group Anagrams*/
vector<vector<string>> groupAnagrams(vector<string>& strs) 
{
    unordered_map<string, vector<string>> res_map;
    for(string& str : strs)
    {
        string temp = str;
        std::sort(temp.begin(), temp.end());
        if(res_map.find(temp) != res_map.end())
            res_map[temp].push_back(str);
        else
        {
            vector<string> vecTemp;
            vecTemp.push_back(str);
            res_map[temp] = vecTemp;
        }
    }

    vector<vector<string>> result;
    for(auto it = res_map.begin(); it != res_map.end(); it++)
       result.push_back(it->second); 

    return result;
}

/********************/
/* 34. Find First and Last Position of Element in Sorted Array*/
int find_first_target_num(vector<int>& nums, int target, int start, int end)
{
    if(start > end)
        return -1;
    int mid = (start + end) / 2;
    if(nums[mid] == target)
    {
        if(mid == 0 || nums[mid - 1] != target)
            return mid;
        else
            end = mid - 1;
    }
    else if(nums[mid] > target)
        end = mid - 1;
    else
        start = mid + 1;
    
    return find_first_target_num(nums, target, start, end);
}

int find_last_target_num(vector<int>& nums, int target, int start, int end)
{
    if(start > end)
        return -1;
    int mid = (start + end) / 2;
    if(nums[mid] == target)
    {
        if(mid == nums.size() - 1 || nums[mid + 1] != target)
            return mid;
        else
            start = mid + 1;
    }
    else if(nums[mid] > target)
        end = mid - 1;
    else
        start = mid + 1;
    
    return find_last_target_num(nums, target, start, end);
}

vector<int> searchRange(vector<int>& nums, int target)
{
    vector<int> res(2, -1);
    if(nums.size() == 0)
    {
        return res;
    }

    int s = find_first_target_num(nums, target, 0, nums.size() - 1);
    int e = find_last_target_num(nums, target, 0, nums.size() - 1);

    res[0] = s;    
    res[1] = e;    
    return res;
}
/********************/
/* 297. Serialize and Deserialize Binary Tree*/
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string result = "";
        if(root == NULL)
        {
            result = "null,";
            return result;
        }

        result += std::to_string(root->val);
        result += ",";

        result += serialize(root->left);
        result += serialize(root->right);

        std::cout << result << endl;
        return result;
    }

    void construct_tree(list<string>& vecNodes, TreeNode **root)
    {
        if(vecNodes.size() == 0)
            return;

        if(vecNodes.front() == "null") 
        {
            vecNodes.pop_front();
            *root = NULL;
            return;
        }

        int val = std::atoi(vecNodes.front().c_str());
        *root = new TreeNode(val);
        vecNodes.pop_front();

        construct_tree(vecNodes, &((*root)->left));
        construct_tree(vecNodes, &((*root)->right));
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        if(data.empty())
            return NULL;

        list<string> vecNodes;
        std::stringstream ss(data);        
        string token;
        while (getline(ss,token, ','))
        {
            vecNodes.push_back(token); 
        }

        if(vecNodes.size() == 0)
            return NULL;
        
        TreeNode *root = NULL;
        construct_tree(vecNodes, &root);

        return root;
    }
};
/********************/
/* 260. Single Number III*/
int find_1bit_index(int i)
{
    int temp = 1;
    int idx = 0;
    while(temp & i == 0)
    {
        temp = temp << 1;
        idx++;
    }

    return idx;
}

bool is_idx_bit_1(int i, int idx)
{
    int temp = 1 << idx;
    return temp & i; 
}

vector<int> singleNumberIII(vector<int>& nums) 
{
    vector<int> result;    
    if(nums.size() == 0)
        return result;
    int temp = 0; 
    for(int i : nums)
        temp ^= i;

    int targetIdx = find_1bit_index(temp);
    int m = 0, n = 0;
    for(int i : nums)     
    {
        if(is_idx_bit_1(i, targetIdx))
            m ^= i;
        else
            n ^= i;
    }

    result.push_back(m);
    result.push_back(n);
    return result;
}
/********************/
/* 137. Single Number II*/
int singleNumberII(vector<int>& nums) 
{
    const int BIT_COUNT = 32;
    int bits[BIT_COUNT];
    for(int i : nums)
    {
        int64_t bitInit = 1;
        for(int j = 0; j < BIT_COUNT; j++)
        {
            bits[j] += (bitInit & i);
            bitInit = bitInit << 1;
        }
    }

    int result = 0;
    for(int i = BIT_COUNT - 1; i >= 0; i--)
    {
        result = result << 1;
        result += bits[i] % 3;
    }

    return result;
}
/********************/
/* 239. Sliding Window Maximum*/
vector<int> maxSlidingWindow(vector<int>& nums, int k)
{
    vector<int> result;
    if(nums.size() == 0)
        return result;
    deque<int> index;    

    for(int i = 0; i < k; i++)
    {
        while(!index.empty() && nums[i] > nums[index.back()])
            index.pop_back();
        index.push_back(i);
    }

    for(int i = k; i < nums.size(); i++)
    {
        result.push_back(nums[index.front()]);

        while(!index.empty() && nums[i] > nums[index.back()])
            index.pop_back();
        if(!index.empty() && index.front() < (i - k))
            index.pop_front();

        index.push_back(i);
    }

    result.push_back(nums[index.front()]);
    return result;
}
/********************/
/* 76. Minimum Window Substring*/
string minWindow(string s, string t) 
{
    if(s.empty() || t.empty())
        return "";

    unordered_map<char, int> dict;
    unordered_map<char, int> window;

    for(char c : t)
        dict[c]++;

    int require = dict.size();
    int form = 0;

    int ans[] = {-1, 0, 0};
    int l = 0, r = 0;

    while(r < s.size())
    {
        char c = s[r];
        window[c]++;

        if(dict.find(c) != dict.end() && dict[c] == window[c])
            form++;

        while(form == require && l <= r)
        {
            int len = r - l + 1;
            if(ans[0] == -1 || len < ans[0])
            {
                ans[0] = len;
                ans[1] = l;
                ans[2] == r;
            }

            char tempLeft = s[l];
            window[tempLeft]--;
            if(dict.find(tempLeft) != dict.end() && window[tempLeft] != dict[tempLeft])
                form--;

            l++;
        }

        r++;
    }

    if(ans[0] == -1)
        return "";
    else
    {
        return s.substr(l, ans[0]);
    }
}
/********************/
/* 139. Word Break*/
bool wordBreak(string s, vector<string>& wordDict)
{
    if(s.empty())
        return false;
    if(wordDict.size() == 0)
        return true;

    bool dp[s.length() + 1] = {0};
    dp[0] = 1;

    for(int i = 1; i <= s.length(); i++)
    {
        for(int j = 0; j < i; j++)
        {
            auto find = std::find(wordDict.begin(), wordDict.end(), s.substr(j, i - j));
            if(dp[j] && find != wordDict.end())
            {
                dp[i] = 1;
                break;
            }
        }
    }

    return dp[s.length()];
}

/********************/
/* 347. Top K Frequent Elements*/
vector<int> topKFrequent(vector<int>& nums, int k) 
{
    unordered_map<int,int> nmap;
    for(int i : nums)
        nmap[i]++;

    vector<int> res;
    priority_queue<pair<int, int> > pq;
    for(auto it = nmap.begin(); it != nmap.end(); it++)
    {
        pq.push(std::make_pair(it->second, it->first));
        if(pq.size() > nmap.size() - k)
        {
            res.push_back(pq.top().second);
            pq.pop();
        }
    }

    return res;
}
/********************/
/* 67. Add Binary*/
string addBinary(string a, string b) 
{
    string s = "";
    int carry = 0, lena = a.length() - 1, lenb = b.length() - 1;

    while(lena >= 0 || lenb >= 0 || carry == 1)
    {
        carry += lena >= 0 ? a[lena--] - '0': 0; 
        carry += lenb >= 0 ? b[lenb--] - '0': 0; 
        s = char((carry % 2) + '0') + s;

        carry = carry / 2;
    }

    return s;
}
/********************/
/* 394. Decode String*/
string decodeString(string s)
{
    string result = "";
    string strNum = "";
    int num = 0;

    stack<char> st;
    int be = -1, e = -1;//first [ and last ]

    for(int i = 0; i < s.length(); i++)
    {
        char c = s[i];
        if('[' == c)
        {
            if(-1 == be)
            {
                if(!strNum.empty())
                    num = std::atoi(strNum.c_str());
                be = i;
            }
            st.push(c);
        }
        else if(']' == c)
        {
            st.pop();
            if(st.empty())
            {
                e = i;
                for(int j = 0; j < num; j++)
                {
                    result += decodeString(s.substr(be + 1, e - be - 1));
                }

                be = -1;
                strNum = "";
                num = 0;
            }
        }
        else
        {
            if(c >= '0' && c <= '9')
                strNum += c;
            else if(-1 != be)
                result += c;
        }
    }

    return result;
}

/********************/
/* 151. Reverse Words in a String*/
bool reverseStr(string& str, int s, int e)
{
    while(s <= e)
    {
        char temp = str[s];
        str[s] = str[e];
        str[e] = temp;

        s++;
        e--;
    }

    return true;
}

string reverseWords(string s) 
{
    reverseStr(s, 0, s.length() - 1); 

    string result;
    int i = 0, j = s.length() - 1;
    while(i <= j)
    {
        if(s[i] == ' ')
            i++;
        if(s[j] == ' ')
            j--;
    }

    if(i > j)
        return "";

    s = s.substr(i, j - i + 1);

    for(int i = 0; i < s.length(); i++)
    {
        char c = s[i];
        result += c;

        while(c == ' ' && s[i + 1] == ' ')
            i++;
    }

    
}
/********************/
/* 493. Reverse Pairs*/
int merge_and_count(vector<int>& nums, int start, int mid, int end)
{
    int cnt = 0;
    int len_left = mid - start + 1;
    int len_right = end - mid;

    int left[len_left], right[len_right];

    int i = mid, j = end;
    while(i >= start && j > mid)
    {
        if(nums[i] > (2 * nums[j]))
        {
            cnt +=  j - mid; 
            i--;
            j = end;//reset j
        }
        else
            j--;
    }

    i = 0, j = 0;
    for(; i <= len_left; i++)
        left[i] = nums[start + i];
    for(; j < len_right; j++)
        right[j] = nums[mid + j + 1];
    
    i = 0, j = 0;
    int k = start;
    while(i < len_left && j < len_right)
    {
        if(left[i] < right[j])
            nums[k++] = left[i++];
        else if(left[i] >= right[j])
            nums[k++] = right[j++];
    }

    while(i < len_left)
        nums[k++] = left[i++];
    while(j < len_right)
        nums[k++] = right[j++];

    return cnt;
}

int partition(vector<int>& nums, int start, int end)
{
    if(start < end) 
    {
        int mid = (start + end) / 2;
        partition(nums, start, mid);
        partition(nums, mid + 1, end);

        count += merge_and_count(nums, start, mid, end);
    }
}

int count = 0;

int reversePairs(vector<int>& nums) 
{
    partition(nums, 0, nums.size() - 1);   
    return count;
}

/********************/
/* 173. Binary Search Tree Iterator*/
class BSTIterator {
    stack<TreeNode*> st;

public:
    BSTIterator(TreeNode* root) {
        push_all(root);
    }
    
    /** @return the next smallest number */
    int next() {
        if(!st.empty()){
            TreeNode *node = st.top();
            st.pop();
            push_all(node->right);
            return node->val;
        }

        return 0;
    }
    
    /** @return whether we have a next smallest number */
    bool hasNext() {
        return !st.empty(); 
    }

private:
    void push_all(TreeNode *node)
    {
        while(NULL != node)
        {
            st.push(node);
            node = node->left;
        }
    }
};
/********************/
/* Find longest common substring*/
int find_longest_comm_substr(const string& str1, const string& str2)
{
    if(str1.empty() || str2.empty())
        return 0;

    int len1 = str1.length(), len2 = str2.length();
    int dp[len1][len2];
    std::memset(dp, 0, sizeof(dp[0][0]) * len1 * len2);
    int max = 0;

    for(int i = 0; i < len1; i++)
        if(str2[0] == str1[i])
            dp[i][0] = 1;

    for(int i = 0; i < len1; i++)
        if(str1[0] == str2[i])
            dp[0][i] = 1;
    
    for(int i = 1; i < len1; i++)
    {
        for(int j = 1; j < len2; j++)
        {
            if(str1[i] == str2[j])
            {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            }

            if(dp[i][j] > max)
                max = dp[i][j];
        }
    }

    return max;
}

/********************/
/* 336. Palindrome Pairs*/
vector<vector<int>> palindromePairs(vector<string>& words) 
{
    
}

/********************/
/* 42. Trapping Rain Water*/
int trap(vector<int>& height) 
{
    //brute force    
    /*
    int size = height.size();
    int ans = 0;

    for(int i = 1; i < size; i++)
    {
        int max_left = 0, max_right = 0;

        for(int j = i; j >= 0; j--)
            max_left = std::max(max_left, height[j]);
        for(int j = i; j < size; j++)
            max_right = std::max(max_right, height[j]);
        
        ans += (std::min(max_left, max_right) - height[i]);
    }

    return ans;*/
    int size = height.size();
    vector<int> left_max(size), right_max(size);
    left_max[0] = height[0];
    for(int i = 1; i < size; i++)
        left_max[i] = std::max(height[i], left_max[i - 1]);

    right_max[size - 1] = height[size - 1];
    for(int i = size - 2; i >= 0; i--)
        right_max[i] = std::max(height[i], right_max[i + 1]);
    
    int ans = 0;
    for(int i = 1; i < size; i++)
        ans += (std::min(left_max[i], right_max[i]) - height[i]);

    return ans;
}

/********************/
/* 146. LRU Cache*/
class LRUCache {
public:
    LRUCache(int capacity) : capacity_(capacity) {
        
    }
    
    int get(int key) {
        auto it = lru_map.find(key);        
        if(it == lru_map.end())
            return -1;

        //放到链表的最前边
        lru_list.emplace_front(it->second->first, it->second->second);
        lru_list.erase(it->second);
        lru_map[key] = lru_list.begin();

        return it->second->second;
    }
    
    void put(int key, int value) {
        auto it = lru_map.find(key);        
        if(it != lru_map.end()){
            //放到链表的最前边
            lru_list.emplace_front(it->second->first, value);
            lru_list.erase(it->second);
        }
        else
        {
            if(capacity_ <= lru_map.size())
            {
                auto it_back = lru_list.back();
                lru_map.erase(it_back.first);
                lru_list.pop_back();
            }

            lru_list.emplace_front(key, value);
            lru_map[key] = lru_list.begin();
        }
        
    }

private:

private:
    typedef list<pair<int,int>> LRUList;
    typedef list<pair<int,int>>::iterator LRUListIterator;
    typedef map<int, list<pair<int,int>>::iterator> LRUMap;

    int capacity_;
    LRUList lru_list;
    LRUMap lru_map;
};
/********************/
/* 62. Unique path*/
int uniquePaths(int m, int n)
{
    /*vector<vector<int>> dp(m, vector<int>(n, 1));

    for(int i = 1; i < m; i++)
    {
        for(int j = 1; j < n; j++)
        {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }

    return dp[m - 1][n - 1];*/
    vector<int> prev(n, 1), cur(n, 1);
    for (int i = 1; i < m; i++)
    {
        for (int j = 1; j < n; j++)
        {
            //cur[j] = prev[j] + cur[j - 1];
            cur[j] += cur[j - 1];
        }
        //std::swap(prev, cur);
    }

    //return prev[n - 1];
    return cur[n - 1];
}
/********************/
/* 75. Sort Colors*/
void sortColors(vector<int>& nums) 
{
    int n = nums.size();
    int p0 = 0, cur = 0, p2 = n - 1;    

    while(cur <= p2)
    {
        if(nums[cur] == 0)
        {
            std::swap(nums[cur], nums[p0]);
            cur++;
            p0++;
        }
        else if(nums[cur] == 2)
        {
            std::swap(nums[cur], nums[p2]);
            p2--;
        }
        else
            cur++;
    }
}

/********************/
/* 50. Pow(x, n)*/
double PowHelper(double x, long long n)
{
    if(0 == n)    
        return 1;
    if(1 == n)
        return x;
    double ans = myPow(x, n >> 1);
    ans *= ans;

    if(n & 0x1)
        ans *= x;

    return ans;
}

double myPow(double x, int n) 
{
    if(0 == x)
        return 0;
    if(0 == n)
        return 1;
    double ans = PowHelper(x, abs(n));
    if(n < 0)
        ans = 1.0 / ans;

    return ans;
}
/********************/
/* 21. Merge Two Sorted Lists*/
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) 
{
    if(NULL == l1)
        return l2;
    if(NULL == l2)
        return l1;
    if(l1->val < l2->val)
    {
        l1->next = mergeTwoLists(l1->next, l2);
        return l1;
    }
    else
    {
        l2->next = mergeTwoLists(l1, l2->next);
        return l2;
    }
}
/********************/
/* 8. String to Integer (atoi)*/
int myAtoi(string str) 
{
    if(str.empty())
        return 0;
    char first_ch = 0;
    bool bNagative = false;
    double ans = 0;
    int i = 0;

    for(; i < str.length(); i++)
    {
        char ch = str[i];
        if(ch == ' ')
            continue;
        if(first_ch == 0) 
            first_ch = ch;
        if(first_ch < '0' || first_ch > '9')
        {
            if(first_ch != '+' && first_ch != '-')
                return 0;
            else 
            {
                if(first_ch == '-')
                    bNagative = true;
                i++;
                break;
            }
        }
        else
            break;
    }
    
    for(; i < str.length(); i++)
    {
        char ch = str[i];

        if(ch >= '0' && ch <= '9')
        {
            double temp = ch - '0';
            ans = ans * 10 + temp;
        }
        else
        {
            break;
        }
    }

    if(ans > INT32_MAX && !bNagative)
        ans = INT32_MAX;
    else if(bNagative)
    {
        ans = -ans < INT32_MIN ? INT32_MIN : -ans;
    }

    return (int)ans;
}
/********************/
/* 145. Binary Tree Postorder Traversal*/
vector<int> postorderTraversal(TreeNode* root) 
{
    vector<int> res;    
    if(NULL == root)
        return res;
    stack<TreeNode*> st;
    st.push(root);

    while(!st.empty())
    {
        TreeNode *temp = st.top();
        res.push_back(temp->val);
        st.pop();

        if(temp->left)
            st.push(temp->left);
        if(temp->right)
            st.push(temp->right);
    }

    std::reverse(res.begin(), res.end());
    return res;
}
/********************/
/* 373. Find K Pairs with Smallest Sums*/
struct compare
{
    bool operator()(pair<int,int>& p1, pair<int, int>& p2)
    {
        return p1.first + p1.second < p2.first + p2.second;
    }
};

vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) 
{
    vector<vector<int>> ans;
    if(nums1.size() == 0 || nums2.size() == 0)
        return ans;

    priority_queue<pair<int, int>, vector<pair<int,int>>, compare> pq; 
    for(int i = 0; i < nums1.size(); i++)
    {
        for(int j = 0; j < nums2.size(); j++)
        {
            int sum = nums1[i] + nums2[j];
            if(pq.size() >= k)
            {
                int sumTop = pq.top().first + pq.top().second;
                if(sum < sumTop)
                {
                    pq.pop();
                }
                else
                    continue;
            }

            pq.emplace(nums1[i], nums2[j]);
        }
    }

    for(int i = 0 ; i < k; i++)
    {
        if(!pq.empty())
        {
            vector<int> tempAns;
            pair<int,int> top = pq.top();
            tempAns.emplace_back(top.first);
            tempAns.emplace_back(top.second);
            ans.emplace_back(tempAns);

            pq.pop();
        }
    }

    return ans;
}
/********************/
/* 446. Arithmetic Slices II - Subsequence*/
int numberOfArithmeticSlices(vector<int>& A){
    int n = A.size(); 
    int ans = 0;
    vector<map<int,int>> dp;

    for(int i = 1; i < n; i++)
    {
        for(int j = 0; j < i; j++)
        {
            int diff = A[i] - A[j];
            int sum = 0;
            if(dp[j].find(diff) != dp[j].end())
                sum = dp[j][diff];
            dp[i][diff] += sum + 1;
            ans += sum;
        }
    }

    return ans;
} 
/********************/
/* 51. N-Queens*/
class Solution2 {
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res;    
        vector<string> nQueen(n, std::string(n, '.'));
        solveNQueens(res, nQueen, 0, n);

        return res;
    }
private:
    void solveNQueens(vector<vector<string>>& res, vector<string>& nQueen, int row, int& n) {
        if(row == n)
        {
            res.push_back(nQueen);
            return;
        }

        for(int i = 0; i < n; i++){
            if(isValid(nQueen, row, i, n)){
                nQueen[row][i] = 'Q';
                solveNQueens(res, nQueen, row + 1, n);
                nQueen[row][i] = '.';
            }
        }
    }

    bool isValid(vector<string>& nQueen, int row, int col, int& n){
        for(int i = 0; i < row; i++)
            if(nQueen[i][col] == 'Q')
                return false;
        for(int i = row - 1, j = col - 1; i >= 0 && j >= 0; j--, i--)
            if(nQueen[i][j] == 'Q')
                return false;
        for(int i = row - 1, j = col + 1; i >= 0 && j < n; j++, i--)
            if(nQueen[i][j] == 'Q')
                return false;
        
        return true;
    }
};
/********************/
/* 572. Subtree of Another Tree*/
bool isSubTreeEqual(TreeNode *s, TreeNode *t){
    if(!s && !t)
        return true;
    else if(!s || !t)
        return false;
    
    if(s->val == t->val)
    {
        return isSubTreeEqual(s->left, t->left) && isSubTreeEqual(s->right, t->right);
    }

    return false;
}

bool travel(TreeNode *s, TreeNode *t){
    return s && (isSubTreeEqual(s, t) || travel(s->left, t) || travel(s->right, t));
}

bool isSubtree(TreeNode* s, TreeNode* t) {
    return travel(s, t);
}

/********************/
/* 240. Search a 2D Matrix II*/
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int row = matrix.size();
    if(0 == row)
        return false;
    int col = matrix[0].size();    
    if(0 == col)
        return false;
    
    int i = 0, j = col - 1;
    while(i < row && j >= 0)
    {
        if(matrix[i][j] == target)
            return true;
        else if(matrix[i][j] > target)
            j--;
        else if(matrix[i][j] < target)
            i++;
    }

    return false;
}
/********************/
/* 31. Next Permutation*/
void nextPermutation(vector<int>& nums) {
    int n = nums.size(), i, j;

    for(i = n - 2; i >= 0; i--) {
        if(nums[i] < nums[i + 1])
            break;
    }
    if(i < 0){
        std::reverse(nums.begin(), nums.end());
    } else{
        for(j = n - 1; j > i; j--)
            if(nums[j] > nums[i])
                break;
        std::swap(nums[j], nums[i]);
        std::reverse(nums.begin() + i + 1, nums.end());
    }
}
/********************/
/* 301. Remove Invalid Parentheses */
void removeInvalidParHelper(string s, int last_i, int last_j, const string& strPair, vector<string>& ans){
    int stack = 0;
    for(int i = last_i; i < s.length(); i++){
        if(s[i] == strPair[0])
            stack++;
        else if(s[i] == strPair[1])
            stack--;
        
        if(stack >= 0)
            continue;
        for(int j = last_j; j <= i; j++) {
            if(s[j] == strPair[1] && (j == last_j || s[j - 1] != strPair[1])) {
                string strRemoved = s.substr(0, j) + s.substr(j + 1, s.length() - j - 1);
                removeInvalidParHelper(strRemoved, i, j, strPair, ans);
            }
        }
        return;
    }

    std::reverse(s.begin(), s.end());

    if(strPair[0] == '(')
        removeInvalidParHelper(s, 0, 0, ")(", ans);
    else
        ans.push_back(s);
}

vector<string> removeInvalidParentheses(string s) {
   vector<string> ans; 
   removeInvalidParHelper(s, 0, 0, "()", ans);
   return ans;
}
/********************/
/* 975. Odd Even Jump*/
//Brute force(TIME LIMIT EXCEEDED)
int find_next(vector<int>& A, int nCur, bool bOdd){
    int targetIndex = -1;

    if(bOdd){ //find smallest bigger
        int targetValue = INT32_MAX;
        for(int i = nCur + 1; i < A.size(); i++){
            if(A[i] >= A[nCur]){
                if(A[i] < targetValue) {
                    targetValue = A[i];
                    targetIndex = i;
                }
            }
        }
    }else{
        int targetValue = INT32_MIN;
        for(int i = nCur + 1; i < A.size(); i++){
            if(A[i] <= A[nCur]){
                if(A[i] > targetValue) {
                    targetValue = A[i];
                    targetIndex = i;
                }
            }
        }
    }

    return targetIndex;
}

int oddEvenJumps(vector<int>& A) {
    /* Brute force(TIME LIMIT EXCEEDED)
    int ans = 0;

    for(size_t i = 0; i < A.size(); i++){

        int nJumpTime = 0;
        int nCur = i;
        while(true){
            nJumpTime++;
            nCur = find_next(A, nCur, nJumpTime & 1);
            if(-1 == nCur)
                break;
            else if(A.size() - 1 == nCur){
                ans++;
                break;
            }
        }
    }

    return ans;*/
    //Use DP
    int ans = 0;
    int n = A.size();
    vector<int> dpHigher(n, 0), dpLower(n, 0);
    map<int, int> mp;

    dpHigher[n - 1] = dpLower[n - 1] = 1;
    mp[A[n - 1]] = n - 1;

    for(int i = n - 2; i >= 0; i--){
        auto hi = mp.lower_bound(A[i]);
        auto lo = mp.upper_bound(A[i]);
        if(hi != mp.end())
            dpHigher[i] = dpLower[hi->second];
        if(lo != mp.begin())
            dpLower[i] = dpHigher[(--lo)->second];
        if(dpHigher[i])
            ans++;
        mp[A[i]] = i;
    }

    return ans;
}
/********************/
/* 85. Maximal Rectangle*/
struct PointInfo
{
    int sum_row;
    int sum_col;
};

int maximalRectangle(vector<vector<char>>& matrix) {
    if(matrix.size() == 0) 
        return 0;

    int row = matrix.size();
    int col = matrix[0].size();
    PointInfo dp[row][col];
    return 0;
}

/********************/
/* 287. Find the Duplicate Number*/
int getRangeCount(vector<int>& nums, int start, int end) {
    if(nums.size() == 0)
        return 0;
    int count = 0;
    for(int i : nums){
        if(i >= start && i <= end)
            count++;
    }

    return count;
}

int findDuplicate(vector<int>& nums) {
    if(0 == nums.size())    
        return 0;
    int n = nums.size();
    int start = 1, end = n - 1;
    while(start <= end){
        int mid = ((end - start) >> 1) + start;
        int countLeft = getRangeCount(nums, start, mid);
        if(start == end){
            if(countLeft > 1)
                return start;
            else
                break;
        }

        if(countLeft > (mid - start + 1))
            end = mid;
        else
            start = mid + 1;
    }

    return - 1;
}
/********************/
/* 91. Decode Ways*/
int numDecodings(string s) {
    if(s.empty())
        return 0;
    int n = s.length();
    vector<int> dp(n + 1, 0);
    dp[0] = 1;
    dp[1] = s[0] == '0' ? 0 : 1;

    for(int i = 2; i <= n; i++){
        if(s[i - 1] > '0' && s[i - 1] <= '9')
            dp[i] += dp[i - 1];
        string temp = s.substr(i - 2, 2);
        int nTemp = std::atoi(temp.c_str());
        if(nTemp >= 10 && nTemp <= 26)
            dp[i] += dp[i - 2];
    }

    return dp[n];
}

/********************/
/* 895. Maximum Frequency Stack*/
class FreqStack {
private:
    unordered_map<int, int> freq;
    unordered_map<int, stack<int>> group;
    int maxFreq;
public:
    FreqStack() {
       maxFreq = 0; 
    }
    
    void push(int x) {
        maxFreq = std::max(maxFreq, ++freq[x]);
        group[freq[x]].push(x);
    }
    
    int pop() {
       int x = group[maxFreq].top(); 
       group[freq[x]--].pop();
       if(group[freq[x]].size() == 0)
            maxFreq--;
       return x;
    }
};

/********************/
/* 26. Remove Duplicates from Sorted Array*/
int removeDuplicates(vector<int>& nums) {
    int n = nums.size();
    if(0 == n)    
        return 0;
    int nLast = nums[0];
    int cur = 1;
    for(int i = 1; i < n; i++) {
        if(nums[i] == nLast){
            continue;
        }else{
            nLast = nums[i];
            std::swap(nums[cur], nums[i]);
            cur++;
        }
    }
    
    return cur;
}

/********************/
/* 127. Word Ladder*/
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    int size = wordList.size();
    if(0 == size)
        return 0;

    unordered_set<string> myset(wordList.begin(), wordList.end());
    queue<string> q;
    q.push(beginWord);
    int ans = 1;

    while(!q.empty()){
        int tempSize = q.size();
        for(int i = 0; i < tempSize; i++){
            string temp = q.front();
            q.pop();

            if(temp == endWord)
                return ans;

            myset.erase(temp);
            
            for(int j = 0; j < temp.size(); j++){
                int c = temp[j];
                for(int k = 0; k < 26; k++){
                    temp[j] = k + 'a';
                    if(myset.find(temp) != myset.end())
                        q.push(temp);
                }
                temp[j] = c;
            }
        }
        ans++;
    }

    return 0;
}
/********************/
/* 341. Flatten Nested List Iterator*/
class NestedInteger; 
class NestedIterator {
public:
/*
    NestedIterator(vector<NestedInteger> &nestedList) {
       for(int i = nestedList.size() - 1; i >= 0; i--) {
           st.push(nestedList[i]);
       }
    }

    int next() {
       if(hasNext()) 
            return st.top().getInteger();
       return NULL;    
    }

    bool hasNext() {
        while(!st.empty()) {
            auto x = st.top();
            if(x.isInteger())
                return true;
            st.pop();
            auto list = x.getList();
            for(int i = list.size() - 1; i >= 0; i--){
                st.push(list[i]);
            }
        }

        reture false;
    }

private:
    stack<NestedInteger> st;*/
};
/********************/
/* 64. Minimum Path Sum*/
int minPathSum(vector<vector<int>>& grid) {
    int row = grid.size();
    if(0 == row)
        return 0;
    int col = grid[0].size();
    if(0 == col)
        return 0;
    vector<int> sumDp(col, 0);
    
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            int up = i ? sumDp[j] : INT_MAX;
            int left = j ? sumDp[j - 1] : INT_MAX;
            if(i == 0 && j == 0) {
                sumDp[0] = grid[0][0];
                continue;
            }
            sumDp[j] = std::min(left, up) + grid[i][j];
        }
    }

    return sumDp[col - 1];
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