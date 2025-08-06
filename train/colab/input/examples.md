/**
 * SL2 Query JS API Training Examples
 * Based on CO2 Query JS API documentation
 * 
 * File for training deepseek-coder model on SL2 Query API patterns
 * Save as: /content/gdrive/MyDrive/Colab Notebooks/input/examples.md
 */

// =================
// BASIC FIND OPERATIONS
// =================

const basicFind = async () => {
    const model = await p.getModel('user');
    
    // Find with simple condition
    const users = await model.find({ active: true });
    
    // Find with multiple conditions (AND)
    const activeAdmins = await model.find({ 
        active: true, 
        role: 'admin' 
    });
    
    // Find with OR conditions
    const adminOrManager = await model
        .find({ role: 'admin' })
        .orFind({ role: 'manager' });
    
    return { users, activeAdmins, adminOrManager };
};

const findWithConditions = async () => {
    const model = await p.getModel('product');
    
    // Complex conditions
    const products = await model
        .find({ 
            category: 'electronics',
            price: { '>': 100 }
        })
        .orFind({ 
            featured: true,
            stock: { '>': 0 }
        });
    
    return products;
};

const findOneExample = async () => {
    const model = await p.getModel('user');
    
    // Find single record
    const admin = await model.findOne({ role: 'admin' });
    
    // Find first match with ordering
    const newestUser = await model
        .find({ active: true })
        .order({ created_at: 'desc' })
        .findOne();
    
    return { admin, newestUser };
};

// =================
// COMPARISON OPERATORS
// =================

const comparisonOperators = async () => {
    const model = await p.getModel('product');
    
    // Greater than, less than
    const expensiveProducts = await model.find({
        price: { '>': 100 }
    });
    
    // Greater than or equal
    const premiumProducts = await model.find({
        rating: { '>=': 4.5 }
    });
    
    // Range (between)
    const midRangeProducts = await model.find({
        price: { '>': 50, '<': 200 }
    });
    
    // Not equal
    const notDiscontinued = await model.find({
        status: { '!=': 'discontinued' }
    });
    
    return { expensiveProducts, premiumProducts, midRangeProducts, notDiscontinued };
};

const inAndNotInOperators = async () => {
    const model = await p.getModel('product');
    
    // IN operator
    const specificCategories = await model.find({
        category: ['electronics', 'books', 'clothing']
    });
    
    // NOTIN operator
    const excludeCategories = await model.find({
        category: { 'NOTIN': ['restricted', 'banned'] }
    });
    
    // Multiple IN conditions
    const filteredProducts = await model.find({
        category: ['electronics', 'software'],
        status: ['active', 'featured']
    });
    
    return { specificCategories, excludeCategories, filteredProducts };
};

// =================
// TEXT SEARCH OPERATORS
// =================

const textSearchOperators = async () => {
    const model = await p.getModel('product');
    
    // STARTSWITH operator
    const appleProducts = await model.find({
        name: { 'STARTSWITH': 'Apple' }
    });
    
    // ENDSWITH operator
    const proProducts = await model.find({
        name: { 'ENDSWITH': 'Pro' }
    });
    
    // LIKE (contains) operator
    const smartphoneProducts = await model.find({
        description: { 'LIKE': 'smartphone' }
    });
    
    // NOTLIKE operator
    const notDiscontinuedProducts = await model.find({
        description: { 'NOTLIKE': 'discontinued' }
    });
    
    return { appleProducts, proProducts, smartphoneProducts, notDiscontinuedProducts };
};

const nullCheckOperators = async () => {
    const model = await p.getModel('user');
    
    // IS NULL
    const usersWithoutDescription = await model.find({
        description: 'ISNULL'
    });
    
    // IS NOT NULL
    const usersWithDescription = await model.find({
        description: 'ISNOTNULL'
    });
    
    // Multiple null checks
    const incompleteProfiles = await model.find({
        bio: 'ISNULL',
        avatar: 'ISNULL'
    });
    
    return { usersWithoutDescription, usersWithDescription, incompleteProfiles };
};

// =================
// ARRAY OPERATORS
// =================

const arrayContainsOperators = async () => {
    const model = await p.getModel('project');
    
    // Array contains one of
    const projectsWithTags = await model.find({
        tags: ['urgent', 'high-priority']
    });
    
    // Array contains (strict) - all specified values
    const projectsWithAllTags = await model.find({
        tags: { '@>': ['javascript', 'frontend'] }
    });
    
    // Array contains one of (explicit syntax)
    const projectsWithAnyTag = await model.find({
        tags: { '@~': ['react', 'vue', 'angular'] }
    });
    
    return { projectsWithTags, projectsWithAllTags, projectsWithAnyTag };
};

const arrayNegationOperators = async () => {
    const model = await p.getModel('project');
    
    // Array does not contain
    const projectsWithoutTags = await model.find({
        tags: { '!@>': ['deprecated', 'legacy'] }
    });
    
    // Array is not contained by
    const projectsNotContainedBy = await model.find({
        tags: { '!<@': ['web', 'mobile'] }
    });
    
    // Array not equal
    const projectsNotExactTags = await model.find({
        tags: { '!=': ['old', 'outdated'] }
    });
    
    return { projectsWithoutTags, projectsNotContainedBy, projectsNotExactTags };
};

const arrayEqualityOperators = async () => {
    const model = await p.getModel('project');
    
    // Array equal to
    const exactTagsProjects = await model.find({
        tags: { '=': ['web', 'mobile'] }
    });
    
    // Array is contained by
    const projectsContainedBy = await model.find({
        tags: { '<@': ['javascript', 'python', 'java', 'web', 'mobile'] }
    });
    
    // Empty array check
    const projectsWithoutTags = await model.find({
        tags: 'ISNULL'
    });
    
    return { exactTagsProjects, projectsContainedBy, projectsWithoutTags };
};

// =================
// RTL QUERIES
// =================

const rtlBasicQueries = async () => {
    const model = await p.getModel('user');
    
    // RTL by IDs
    const usersInGroups = await model.find({
        user_groups: [1, 2, 3]
    });
    
    // RTL NOT IN
    const usersNotInGroups = await model.find({
        user_groups: { NOTIN: [4, 5, 6] }
    });
    
    // RTL by alias
    const usersByGroupAlias = await model.find({
        user_groups: { alias: ['admin', 'manager'] }
    });
    
    return { usersInGroups, usersNotInGroups, usersByGroupAlias };
};

const rtlAdvancedQueries = async () => {
    const model = await p.getModel('employee');
    
    // RTL with multiple aliases
    const employeesInDepartments = await model.find({
        departments: { alias: ['sales', 'marketing', 'development'] }
    });
    
    // RTL NOTIN with aliases
    const employeesNotInDepartments = await model.find({
        departments: { alias: { NOTIN: ['hr', 'finance'] } }
    });
    
    // RTL complex conditions
    const activeEmployeesInTechDepts = await model.find({
        active: true,
        departments: { alias: ['development', 'qa', 'devops'] }
    });
    
    return { employeesInDepartments, employeesNotInDepartments, activeEmployeesInTechDepts };
};

// =================
// REFERENCE QUERIES
// =================

const referenceQueries = async () => {
    const model = await p.getModel('document');
    
    // Reference by ID
    const documentsByModel = await model.find({
        model: 1
    });
    
    // Reference NOT IN
    const documentsNotInModels = await model.find({
        model: { NOTIN: [1, 2, 3] }
    });
    
    // Reference by alias
    const userDocuments = await model.find({
        model: { alias: 'user' }
    });
    
    return { documentsByModel, documentsNotInModels, userDocuments };
};

const referenceAdvancedQueries = async () => {
    const model = await p.getModel('permission');
    
    // Multiple reference aliases
    const permissionsForModels = await model.find({
        model: { alias: ['user', 'document', 'project'] }
    });
    
    // Reference NOTIN with aliases
    const permissionsNotForModels = await model.find({
        model: { alias: { NOTIN: ['system', 'internal'] } }
    });
    
    // Reference with other conditions
    const activePermissions = await model.find({
        active: true,
        model: { alias: 'user' }
    });
    
    return { permissionsForModels, permissionsNotForModels, activePermissions };
};

// =================
// GLOBALREF QUERIES
// =================

const globalRefQueries = async () => {
    const model = await p.getModel('attachment');
    
    // GlobalRef exact match
    const attachmentsForRecord = await model.find({
        global_ref: { model: 33, id: 674 }
    });
    
    // GlobalRef not equal
    const attachmentsNotForRecord = await model.find({
        global_ref: { '!=': { model: 33, id: 674 } }
    });
    
    // GlobalRef with other conditions
    const activeAttachments = await model.find({
        active: true,
        global_ref: { model: 50, id: 123 }
    });
    
    return { attachmentsForRecord, attachmentsNotForRecord, activeAttachments };
};

const globalRefAdvancedQueries = async () => {
    const model = await p.getModel('comment');
    
    // GlobalRef for specific model type
    const userComments = await model.find({
        global_ref: { model: 1 }  // All records from model 1
    });
    
    // GlobalRef null check
    const orphanedComments = await model.find({
        global_ref: 'ISNULL'
    });
    
    // GlobalRef not null
    const linkedComments = await model.find({
        global_ref: 'ISNOTNULL'
    });
    
    return { userComments, orphanedComments, linkedComments };
};

// =================
// ORDERING AND SORTING
// =================

const basicOrdering = async () => {
    const model = await p.getModel('article');
    
    // Simple ascending order
    const articlesByTitle = await model
        .find({ published: true })
        .order({ title: 'asc' });
    
    // Simple descending order
    const articlesByDate = await model
        .find({ published: true })
        .order({ created_at: 'desc' });
    
    // Multiple field ordering
    const articlesByPriorityAndDate = await model
        .find({ status: 'active' })
        .order({ priority: 'desc', created_at: 'asc' });
    
    return { articlesByTitle, articlesByDate, articlesByPriorityAndDate };
};

const advancedOrdering = async () => {
    const model = await p.getModel('product');
    
    // Order by calculated fields
    const productsByPopularity = await model
        .find({ active: true })
        .order({ 
            featured: 'desc',
            rating: 'desc', 
            price: 'asc' 
        });
    
    // Order with conditions
    const sortedProducts = await model
        .find({ category: 'electronics' })
        .orFind({ category: 'software' })
        .order({ 
            category: 'asc',
            name: 'asc' 
        });
    
    return { productsByPopularity, sortedProducts };
};

// =================
// FIELD SELECTION
// =================

const basicFieldSelection = async () => {
    const model = await p.getModel('user');
    
    // Select specific fields
    const userNames = await model
        .find({ active: true })
        .fields(['id', 'name', 'email']);
    
    // Select fields with conditions
    const adminInfo = await model
        .find({ role: 'admin' })
        .fields(['id', 'name', 'email', 'last_login']);
    
    // Minimal field selection
    const userIds = await model
        .find({ active: true })
        .fields(['id']);
    
    return { userNames, adminInfo, userIds };
};

const advancedFieldSelection = async () => {
    const model = await p.getModel('product');
    
    // Fields with ordering
    const productSummary = await model
        .find({ active: true })
        .fields(['id', 'name', 'price', 'rating'])
        .order({ rating: 'desc' })
        .limit(10);
    
    // Fields with complex conditions
    const featuredProductInfo = await model
        .find({ featured: true })
        .orFind({ rating: { '>=': 4.5 } })
        .fields(['id', 'name', 'category', 'price'])
        .order({ price: 'desc' });
    
    return { productSummary, featuredProductInfo };
};

// =================
// RAW DATA QUERIES
// =================

const basicRawQueries = async () => {
    const model = await p.getModel('stats');
    
    // Simple raw query
    const rawStats = await model
        .find({ date: { '>': new Date('2024-01-01') } })
        .raw();
    
    // Raw with field selection
    const rawUserData = await model
        .find({ active: true })
        .fields(['id', 'name', 'created_at'])
        .raw();
    
    // Raw with ordering
    const rawOrderedData = await model
        .find({ status: 'completed' })
        .order({ created_at: 'desc' })
        .raw();
    
    return { rawStats, rawUserData, rawOrderedData };
};

const advancedRawQueries = async () => {
    const model = await p.getModel('analytics');
    
    // Raw data for processing
    const processingData = await model
        .find({ 
            processed: false,
            created_at: { '>': new Date('2024-01-01') }
        })
        .fields(['id', 'data', 'type'])
        .order({ created_at: 'asc' })
        .raw();
    
    // Raw data with complex conditions
    const reportData = await model
        .find({ report_type: 'monthly' })
        .orFind({ report_type: 'quarterly' })
        .fields(['id', 'data', 'period', 'metrics'])
        .raw();
    
    return { processingData, reportData };
};

// =================
// PAGINATION AND LIMITS
// =================

const basicPagination = async () => {
    const model = await p.getModel('article');
    
    // First page (10 records)
    const firstPage = await model
        .find({ published: true })
        .order({ created_at: 'desc' })
        .limit(10, 0);
    
    // Second page (next 10 records)
    const secondPage = await model
        .find({ published: true })
        .order({ created_at: 'desc' })
        .limit(10, 10);
    
    // Just limit without offset
    const topTen = await model
        .find({ featured: true })
        .order({ views: 'desc' })
        .limit(10);
    
    return { firstPage, secondPage, topTen };
};

const advancedPagination = async () => {
    const model = await p.getModel('product');
    
    // Pagination with complex conditions
    const paginatedProducts = await model
        .find({ category: 'electronics' })
        .orFind({ featured: true })
        .order({ rating: 'desc', price: 'asc' })
        .limit(20, 40); // Page 3, 20 per page
    
    // Large dataset pagination
    const batchData = await model
        .find({ needs_processing: true })
        .order({ created_at: 'asc' })
        .limit(1000, 0);
    
    return { paginatedProducts, batchData };
};

// =================
// COUNT OPERATIONS
// =================

const basicCountOperations = async () => {
    const model = await p.getModel('user');
    
    // Simple count
    const totalUsers = await model
        .find({ active: true })
        .count();
    
    // Count with conditions
    const adminCount = await model
        .find({ role: 'admin' })
        .count();
    
    // Count with OR conditions
    const privilegedUsersCount = await model
        .find({ role: 'admin' })
        .orFind({ role: 'manager' })
        .count();
    
    return { totalUsers, adminCount, privilegedUsersCount };
};

const advancedCountOperations = async () => {
    const model = await p.getModel('order');
    
    // Count with complex conditions
    const recentOrdersCount = await model
        .find({ 
            status: 'completed',
            created_at: { '>': new Date('2024-01-01') }
        })
        .count();
    
    // Count with array conditions
    const urgentOrdersCount = await model
        .find({ 
            priority: ['high', 'urgent'],
            status: { '!=': 'cancelled' }
        })
        .count();
    
    return { recentOrdersCount, urgentOrdersCount };
};

// =================
// GROUPING OPERATIONS
// =================

const basicGrouping = async () => {
    const model = await p.getModel('sales');
    
    // MAX aggregation
    const maxSalesByCategory = await model
        .find({ status: 'completed' })
        .group('category', { 'max_amount': { 'MAX': 'amount' } });
    
    // MIN aggregation
    const minSalesByCategory = await model
        .find({ status: 'completed' })
        .group('category', { 'min_amount': { 'MIN': 'amount' } });
    
    // SUM aggregation
    const totalSalesByCategory = await model
        .find({ status: 'completed' })
        .group('category', { 'total_amount': { 'SUM': 'amount' } });
    
    return { maxSalesByCategory, minSalesByCategory, totalSalesByCategory };
};

const advancedGrouping = async () => {
    const model = await p.getModel('analytics');
    
    // Multiple aggregations
    const salesStats = await model
        .find({ status: 'completed' })
        .group('region', {
            'total_sales': { 'SUM': 'amount' },
            'avg_sale': { 'AVG': 'amount' },
            'sale_count': { 'COUNT': 'id' },
            'max_sale': { 'MAX': 'amount' },
            'min_sale': { 'MIN': 'amount' }
        });
    
    // First and Last aggregations
    const timeStats = await model
        .find({ type: 'transaction' })
        .order({ created_at: 'asc' })
        .group('category', {
            'first_transaction': { 'FIRST': 'amount' },
            'last_transaction': { 'LAST': 'amount' }
        });
    
    return { salesStats, timeStats };
};

// =================
// BULK OPERATIONS
// =================

const bulkUpdateOperations = async () => {
    const model = await p.getModel('user');
    
    // Basic bulk update
    const updatedCount = await model
        .find({ last_login: { '<': new Date('2023-01-01') } })
        .update({ status: 'inactive' });
    
    // Bulk update with multiple fields
    const archiveCount = await model
        .find({ 
            status: 'inactive',
            last_login: { '<': new Date('2022-01-01') }
        })
        .update({ 
            status: 'archived',
            archived_at: new Date(),
            active: false
        });
    
    return { updatedCount, archiveCount };
};

const bulkUpdateWithSQL = async () => {
    const model = await p.getModel('product');
    
    // Bulk update with SQL raw values
    const updatedWithSQL = await model
        .find({ id: [1, 2, 3] })
        .update({
            name: { 
                __type: 'sql_raw', 
                sql: `CONCAT('Product_', id)` 
            },
            status: { 
                __type: 'sql_raw', 
                sql: `(CASE WHEN id = 1 THEN ? WHEN id = 2 THEN ? ELSE ? END)`, 
                bindings: ['active', 'pending', 'inactive'] 
            }
        });
    
    // Complex SQL update
    const complexUpdate = await model
        .find({ category: 'electronics' })
        .update({
            price: {
                __type: 'sql_raw',
                sql: `price * 1.1`  // 10% price increase
            }
        });
    
    return { updatedWithSQL, complexUpdate };
};

const bulkDeleteOperations = async () => {
    const model = await p.getModel('log');
    
    // Basic bulk delete
    const deletedCount = await model
        .find({ 
            created_at: { '<': new Date('2023-01-01') },
            level: 'debug'
        })
        .delete();
    
    // Bulk delete with complex conditions
    const cleanupCount = await model
        .find({ processed: true })
        .orFind({ 
            created_at: { '<': new Date('2024-01-01') },
            level: { 'NOTIN': ['error', 'warning'] }
        })
        .delete();
    
    return { deletedCount, cleanupCount };
};

const massInsertOperations = async () => {
    const model = await p.getModel('user');
    
    // Basic mass insert
    const insertedIds = await model.massInsert([
        { name: 'John Doe', email: 'john@example.com', active: true },
        { name: 'Jane Smith', email: 'jane@example.com', active: true },
        { name: 'Bob Johnson', email: 'bob@example.com', active: false }
    ]);
    
    // Mass insert with complex data
    const productIds = await model.massInsert([
        { 
            name: 'Laptop Pro', 
            category: 'electronics', 
            price: 1299.99,
            features: ['high-performance', 'portable']
        },
        { 
            name: 'Wireless Mouse', 
            category: 'accessories', 
            price: 29.99,
            features: ['wireless', 'ergonomic']
        }
    ]);
    
    return { insertedIds, productIds };
};

// =================
// RTL BULK OPERATIONS
// =================

const rtlBulkOperations = async () => {
    const model = await p.getModel('user');
    
    // RTL replace operation
    const replacedCount = await model
        .find({ id: [1, 2, 3] })
        .update({
            user_groups: [5, 6, 7] // Replace existing RTL values
        });
    
    // RTL append operation
    const appendedCount = await model
        .find({ active: true })
        .update({
            user_groups: { append: [8, 9] } // Add to existing RTL
        });
    
    // RTL remove operation
    const removedCount = await model
        .find({ status: 'inactive' })
        .update({
            user_groups: { remove: [1, 2] } // Remove from RTL
        });
    
    return { replacedCount, appendedCount, removedCount };
};

const rtlAdvancedBulkOperations = async () => {
    const model = await p.getModel('project');
    
    // RTL operations with conditions
    const teamUpdate = await model
        .find({ 
            status: 'active',
            team_members: { '@>': [1, 2] } // Has specific members
        })
        .update({
            team_members: { append: [10, 11] } // Add new members
        });
    
    // RTL cleanup operation
    const cleanupUpdate = await model
        .find({ status: 'completed' })
        .update({
            assigned_users: { remove: [] }, // Remove all assignments
            tags: { append: ['completed'] }  // Add completion tag
        });
    
    return { teamUpdate, cleanupUpdate };
};

// =================
// JOINS
// =================

const basicJoins = async () => {
    const userModel = await p.getModel('user');
    const profileModel = await p.getModel('profile');
    
    // Simple join
    const usersWithProfiles = await userModel
        .join(profileModel, 'user_id', userModel, 'id')
        .find({ active: true }, userModel)
        .order({ name: 'asc' }, userModel);
    
    // Join with field selection
    const userProfileData = await userModel
        .join(profileModel, 'user_id', userModel, 'id')
        .find({ active: true }, userModel)
        .fields(['id', 'name', 'email'], userModel)
        .fields(['bio', 'avatar'], profileModel);
    
    return { usersWithProfiles, userProfileData };
};

const advancedJoins = async () => {
    const orderModel = await p.getModel('order');
    const customerModel = await p.getModel('customer');
    const productModel = await p.getModel('product');
    
    // Multiple joins
    const orderDetails = await orderModel
        .join(customerModel, 'id', orderModel, 'customer_id')
        .join(productModel, 'id', orderModel, 'product_id')
        .find({ 
            status: 'completed',
            created_at: { '>': new Date('2024-01-01') }
        }, orderModel)
        .orFind({ premium: true }, customerModel)
        .fields(['id', 'total', 'created_at'], orderModel)
        .fields(['name', 'email'], customerModel)
        .fields(['name', 'category'], productModel)
        .order({ total: 'desc' }, orderModel);
    
    return { orderDetails };
};

const joinWithGrouping = async () => {
    const salesModel = await p.getModel('sales');
    const customerModel = await p.getModel('customer');
    
    // Join with grouping
    const customerSalesStats = await salesModel
        .join(customerModel, 'id', salesModel, 'customer_id')
        .find({ status: 'completed' }, salesModel)
        .group('customer_type', {
            'total_sales': { 'SUM': { 'amount': salesModel } },
            'avg_sale': { 'AVG': { 'amount': salesModel } },
            'sale_count': { 'COUNT': { 'id': salesModel } }
        }, customerModel);
    
    return { customerSalesStats };
};

// =================
// DATE TIME OPERATIONS
// =================

const dateTimeBasicQueries = async () => {
    const model = await p.getModel('event');
    
    // Basic date comparisons
    const recentEvents = await model
        .find({ 
            created_at: { '>': new Date('2024-01-01') }
        });
    
    // Date range queries
    const thisMonthEvents = await model
        .find({
            created_at: { 
                '>=': new Date('2024-08-01'),
                '<': new Date('2024-09-01')
            }
        });
    
    return { recentEvents, thisMonthEvents };
};

const dateTimePrecisionQueries = async () => {
    const model = await p.getModel('log');
    
    // Day precision
    const dailyLogs = await model
        .find({ 
            created_at: { '>': new Date('2024-01-01') }
        })
        .dateTrunc('day');
    
    // Hour precision  
    const hourlyStats = await model
        .find({ level: 'error' })
        .dateTrunc('hour');
    
    // Month precision
    const monthlyAnalytics = await model
        .find({ processed: true })
        .dateTrunc('month');
    
    return { dailyLogs, hourlyStats, monthlyAnalytics };
};

// =================
// ITERATOR OPERATIONS
// =================

const iteratorEachExamples = async () => {
    const model = await p.getModel('large_dataset');
    const results = [];
    
    // Process records one by one
    await p.iterEach(
        model
            .find({ processed: false })
            .order({ id: 'asc' })
            .fields(['id', 'data']),
        1000, // batch size
        (record) => {
            // Process each record
            results.push(`Processed record ${record.id}`);
            // Add your processing logic here
        }
    );
    
    return { processedCount: results.length };
};

const iteratorMapExamples = async () => {
    const model = await p.getModel('analytics');
    
    // Map over records and transform data
    const transformedData = await p.iterMap(
        model
            .find({ status: 'active' })
            .fields(['id', 'value', 'multiplier']),
        500, // batch size
        (record) => {
            return {
                id: record.id,
                originalValue: record.value,
                calculatedValue: record.value * (record.multiplier || 1),
                processed: true
            };
        }
    );
    
    return { transformedData };
};

const iteratorFeedExamples = async () => {
    const model = await p.getModel('metrics');
    let totalSum = 0;
    let processedBatches = 0;
    
    // Process batches of records
    await p.iterFeed(
        model
            .find({ type: 'numeric' })
            .fields(['value', 'weight']),
        1000, // batch size
        (records) => {
            // Process entire batch at once
            const batchSum = records.reduce((sum, record) => {
                return sum + (record.value * (record.weight || 1));
            }, 0);
            
            totalSum += batchSum;
            processedBatches++;
        }
    );
    
    return { totalSum, processedBatches };
};

// =================
// COMPLEX QUERIES
// =================

const complexMultiTableQuery = async () => {
    const orderModel = await p.getModel('order');
    const customerModel = await p.getModel('customer');
    const productModel = await p.getModel('product');
    const categoryModel = await p.getModel('category');
    
    // Complex query with multiple joins, conditions, grouping, and ordering
    const salesReport = await orderModel
        .join(customerModel, 'id', orderModel, 'customer_id')
        .join(productModel, 'id', orderModel, 'product_id')
        .join(categoryModel, 'id', productModel, 'category_id')
        .find({ 
            status: 'completed',
            created_at: { '>': new Date('2024-01-01') }
        }, orderModel)
        .orFind({ 
            premium: true,
            active: true
        }, customerModel)
        .fields(['id', 'total', 'created_at'], orderModel)
        .fields(['name', 'email', 'type'], customerModel)
        .fields(['name', 'price'], productModel)
        .fields(['name'], categoryModel)
        .order({ total: 'desc' }, orderModel)
        .order({ name: 'asc' }, customerModel)
        .limit(100);
    
    return { salesReport };
};

const complexGroupingQuery = async () => {
    const salesModel = await p.getModel('sales');
    const regionModel = await p.getModel('region');
    
    // Complex grouping with joins
    const regionalStats = await salesModel
        .join(regionModel, 'id', salesModel, 'region_id')
        .find({ 
            status: 'completed',
            created_at: { '>': new Date('2024-01-01') }
        }, salesModel)
        .group('name', {
            'total_revenue': { 'SUM': { 'amount': salesModel } },
            'avg_sale': { 'AVG': { 'amount': salesModel } },
            'max_sale': { 'MAX': { 'amount': salesModel } },
            'min_sale': { 'MIN': { 'amount': salesModel } },
            'sale_count': { 'COUNT': { 'id': salesModel } },
            'first_sale': { 'FIRST': { 'amount': salesModel } },
            'last_sale': { 'LAST': { 'amount': salesModel } }
        }, regionModel)
        .order({ 'total_revenue': 'desc' });
    
    return { regionalStats };
};

const complexFilteringQuery = async () => {
    const projectModel = await p.getModel('project');
    
    // Complex filtering with multiple OR conditions
    const filteredProjects = await projectModel
        .find({
            status: 'active',
            priority: ['high', 'urgent']
        })
        .orFind({
            featured: true,
            created_at: { '>': new Date('2024-06-01') }
        })
        .orFind({
            tags: { '@>': ['critical', 'important'] },
            assigned_to: 'ISNOTNULL'
        })
        .fields(['id', 'name', 'status', 'priority', 'tags'])
        .order({ priority: 'desc', created_at: 'desc' })
        .limit(50);
    
    return { filteredProjects };
};

// =================
// PERFORMANCE OPTIMIZATIONS
// =================

const optimizedQueries = async () => {
    const model = await p.getModel('large_table');
    
    // Use raw for better performance with large datasets
    const rawData = await model
        .find({ status: 'active' })
        .fields(['id', 'name', 'key_field'])
        .order({ id: 'asc' })
        .limit(1000)
        .raw();
    
    // Optimized field selection for performance
    const minimalData = await model
        .find({ processed: false })
        .fields(['id']) // Only select what you need
        .raw();
    
    return { rawDataCount: rawData.length, minimalDataCount: minimalData.length };
};

const optimizedCounting = async () => {
    const model = await p.getModel('analytics');
    
    // Fast counting without data retrieval
    const totalRecords = await model
        .find({ active: true })
        .count();
    
    // Count with complex conditions
    const filteredCount = await model
        .find({ 
            status: ['active', 'pending'],
            created_at: { '>': new Date('2024-01-01') }
        })
        .orFind({
            priority: 'high',
            processed: false
        })
        .count();
    
    return { totalRecords, filteredCount };
};

const batchProcessingOptimized = async () => {
    const model = await p.getModel('queue');
    let processedCount = 0;
    
    // Efficient batch processing with iterators
    await p.iterEach(
        model
            .find({ status: 'pending' })
            .fields(['id', 'data'])
            .order({ priority: 'desc', created_at: 'asc' }),
        500, // Optimized batch size
        (record) => {
            // Process individual record efficiently
            processedCount++;
            // Your processing logic here
        }
    );
    
    return { processedCount };
};

// =================
// ERROR HANDLING PATTERNS
// =================

const errorHandlingBasic = async () => {
    const model = await p.getModel('user');
    const results = {
        success: [],
        errors: []
    };
    
    try {
        const users = await model.find({ active: true });
        results.success.push(`Found ${users.length} active users`);
        return { users, results };
    } catch (error) {
        results.errors.push(`Query failed: ${error.message}`);
        p.log.error(error, 'User query failed');
        return { users: [], results };
    }
};

const errorHandlingAdvanced = async () => {
    const model = await p.getModel('complex_data');
    let results = [];
    
    // Graceful error handling with fallbacks
    try {
        // Try complex query first
        results = await model
            .find({ 
                complex_field: { 'CUSTOM': 'value' },
                nested_data: { '@>': ['specific', 'values'] }
            })
            .order({ calculated_field: 'desc' });
            
    } catch (complexError) {
        p.log.warn(`Complex query failed: ${complexError.message}. Trying fallback.`);
        
        try {
            // Fallback to simpler query
            results = await model
                .find({ status: 'active' })
                .order({ created_at: 'desc' })
                .limit(100);
                
        } catch (fallbackError) {
            p.log.error(fallbackError, 'Fallback query also failed');
            
            // Final fallback - return empty with error logged
            results = [];
        }
    }
    
    return { results, resultCount: results.length };
};

const errorHandlingBulkOperations = async () => {
    const model = await p.getModel('batch_data');
    const results = {
        updated: 0,
        errors: []
    };
    
    try {
        // Attempt bulk update
        const updatedCount = await model
            .find({ status: 'pending' })
            .update({ 
                status: 'processed',
                processed_at: new Date()
            });
            
        results.updated = updatedCount;
        
    } catch (bulkError) {
        results.errors.push(`Bulk update failed: ${bulkError.message}`);
        p.log.error(bulkError, 'Bulk operation failed');
        
        // Could implement record-by-record processing as fallback
        // This is more complex but provides better error isolation
    }
    
    return results;
};

// =================
// ADVANCED SQL OPERATIONS
// =================

const sqlCasesOperations = async () => {
    const model = await p.getModel('product');
    
    // Prepare data for SQL cases
    const items = [
        { id: 1, name: 'Product A', category: 'electronics', price: 299.99 },
        { id: 2, name: 'Product B', category: 'books', price: 19.99 },
        { id: 3, name: 'Product C', category: 'clothing', price: 49.99 }
    ];
    
    try {
        // Build SQL cases for bulk update
        const sqlCases = await p.utils.buildSqlCasesQuery(items, {
            key: 'id',
            model: 'product'
        });
        
        // Apply bulk update with SQL cases
        const updatedCount = await model
            .find({ id: [1, 2, 3] })
            .update(sqlCases);
        
        return { sqlCases, updatedCount };
        
    } catch (error) {
        p.log.error(error, 'SQL Cases operation failed');
        return { error: error.message };
    }
};

const rawSQLOperations = async () => {
    const model = await p.getModel('analytics');
    
    // Raw SQL in bulk updates
    const updatedWithRawSQL = await model
        .find({ category: 'electronics' })
        .update({
            // Concatenate strings
            display_name: {
                __type: 'sql_raw',
                sql: `CONCAT(category, ' - ', name)`
            },
            // Mathematical operations
            discounted_price: {
                __type: 'sql_raw', 
                sql: `price * 0.9`  // 10% discount
            },
            // Conditional updates
            status: {
                __type: 'sql_raw',
                sql: `(CASE WHEN price > 100 THEN ? WHEN price > 50 THEN ? ELSE ? END)`,
                bindings: ['premium', 'standard', 'basic']
            }
        });
    
    return { updatedWithRawSQL };
};

const conditionalSQLOperations = async () => {
    const model = await p.getModel('user');
    
    // Complex conditional SQL updates
    const userUpdates = await model
        .find({ active: true })
        .update({
            user_level: {
                __type: 'sql_raw',
                sql: `(CASE 
                    WHEN login_count > 100 THEN ? 
                    WHEN login_count > 50 THEN ? 
                    WHEN login_count > 10 THEN ? 
                    ELSE ? 
                END)`,
                bindings: ['expert', 'advanced', 'intermediate', 'beginner']
            },
            last_activity_summary: {
                __type: 'sql_raw',
                sql: `CONCAT('Last login: ', COALESCE(last_login, 'Never'), ' - Status: ', status)`
            }
        });
    
    return { userUpdates };
};

// =================
// UTILITY AND HELPER FUNCTIONS
// =================

const utilityQueryHelpers = async () => {
    const model = await p.getModel('system_data');
    
    // Get SQL query for debugging
    const debugQuery = await model
        .find({ active: true })
        .order({ created_at: 'desc' })
        .limit(10);
    
    const sqlQuery = model.getSQL();
    
    // Disable permissions for system operations
    const systemData = await model
        .find({ system_flag: true })
        .setOptions({ sqlDebug: false, disablePermissions: true });
    
    return { 
        debugQuery, 
        sqlQuery, 
        systemData,
        queryCount: debugQuery.length 
    };
};

const dataValidationHelpers = async () => {
    const model = await p.getModel('validation_data');
    
    // Validate data before processing
    const validRecords = await model
        .find({ 
            email: 'ISNOTNULL',
            email: { 'LIKE': '@' }, // Basic email validation
            status: ['active', 'pending']
        })
        .fields(['id', 'email', 'status']);
    
    // Find incomplete records
    const incompleteRecords = await model
        .find({ email: 'ISNULL' })
        .orFind({ status: 'ISNULL' })
        .fields(['id', 'email', 'status', 'created_at']);
    
    return { 
        validCount: validRecords.length,
        incompleteCount: incompleteRecords.length,
        validRecords,
        incompleteRecords
    };
};

const performanceMonitoringHelpers = async () => {
    const model = await p.getModel('performance_data');
    
    // Monitor query performance
    const startTime = Date.now();
    
    const results = await model
        .find({ 
            processed: false,
            created_at: { '>': new Date(Date.now() - 24 * 60 * 60 * 1000) } // Last 24 hours
        })
        .order({ priority: 'desc' })
        .limit(1000);
    
    const queryTime = Date.now() - startTime;
    
    // Log performance metrics
    p.log.info({
        queryTime,
        resultCount: results.length,
        avgTimePerRecord: queryTime / results.length
    }, 'Query Performance');
    
    return { 
        results, 
        performance: {
            queryTime,
            resultCount: results.length,
            avgTimePerRecord: results.length > 0 ? queryTime / results.length : 0
        }
    };
};

// =================
// INTEGRATION PATTERNS
// =================

const integrationWithForms = async () => {
    // Example of using queries in form context
    const currentRecord = p.record;
    const model = await p.getModel(currentRecord.getModel().getValue('alias'));
    
    // Find related records based on current form data
    const relatedRecords = await model
        .find({ 
            category: currentRecord.getValue('category'),
            id: { '!=': currentRecord.getValue('id') }
        })
        .order({ created_at: 'desc' })
        .limit(5);
    
    // Find records for dropdown population
    const dropdownOptions = await model
        .find({ 
            active: true,
            type: currentRecord.getValue('type')
        })
        .fields(['id', 'name'])
        .order({ name: 'asc' });
    
    return { relatedRecords, dropdownOptions };
};

const integrationWithActions = async () => {
    // Example of using queries in action context
    const requestParams = p.getRequest();
    const model = await p.getModel(requestParams.modelAlias);
    
    // Process selected records from view
    const selectedIds = requestParams.selectedIds || [];
    
    if (selectedIds.length > 0) {
        // Bulk operation on selected records
        const updatedCount = await model
            .find({ id: selectedIds })
            .update({ 
                processed: true,
                processed_at: new Date(),
                processed_by: p.currentUser.getValue('id')
            });
        
        // Log the operation
        p.log.info({
            action: 'bulk_process',
            recordCount: updatedCount,
            userId: p.currentUser.getValue('id')
        }, 'Bulk Processing Completed');
        
        return { updatedCount };
    }
    
    return { updatedCount: 0 };
};

const integrationWithWebServices = async () => {
    // Example of using queries in web service context
    const params = p.getRequest();
    const model = await p.getModel(params.modelAlias);
    
    // API endpoint for data retrieval
    const apiData = await model
        .find({ 
            active: true,
            api_enabled: true
        })
        .fields(['id', 'name', 'api_key', 'endpoint'])
        .order({ name: 'asc' });
    
    // Format for API response
    const formattedData = apiData.map(record => ({
        id: record.getValue('id'),
        name: record.getValue('name'),
        endpoint: record.getValue('endpoint'),
        // Don't expose sensitive data like api_key in public API
    }));
    
    return {
        success: true,
        data: formattedData,
        total: formattedData.length,
        timestamp: new Date().toISOString()
    };
};

// =================
// TESTING AND DEBUGGING
// =================

const testingQueryPatterns = async () => {
    const models = ['user', 'product', 'order', 'customer'];
    const testResults = {};
    
    for (const modelName of models) {
        try {
            const model = await p.getModel(modelName);
            
            // Test basic operations
            const count = await model.find({}).count();
            const sample = await model.find({}).limit(1).raw();
            
            // Test query performance
            const startTime = Date.now();
            const records = await model
                .find({ active: true })
                .limit(100);
            const queryTime = Date.now() - startTime;
            
            testResults[modelName] = {
                totalRecords: count,
                sampleRecord: sample[0] || null,
                queryTime,
                status: 'success'
            };
            
        } catch (error) {
            testResults[modelName] = {
                status: 'error',
                error: error.message
            };
        }
    }
    
    return testResults;
};

const debuggingQueryHelpers = async () => {
    const model = await p.getModel('debug_data');
    
    // Enable SQL debugging
    const debugResults = await model
        .find({ test_field: 'debug_value' })
        .setOptions({ sqlDebug: true });
    
    const sqlQuery = model.getSQL();
    
    // Test different query patterns
    const patterns = {
        simple: await model.find({ id: 1 }).count(),
        complex: await model
            .find({ field1: 'value1' })
            .orFind({ field2: 'value2' })
            .count(),
        withJoin: await model
            .join(await p.getModel('related'), 'id', model, 'related_id')
            .find({ active: true })
            .count()
    };
    
    return {
        debugResults: debugResults.length,
        sqlQuery,
        patterns,
        debugging: true
    };
};

// Main execution function for testing
const executeExamples = async () => {
    try {
        // Test a few key functions
        const basicResults = await basicFind();
        const operatorResults = await comparisonOperators();
        const countResults = await basicCountOperations();
        
        p.log.info({
            basicResults: basicResults ? 'success' : 'failed',
            operatorResults: operatorResults ? 'success' : 'failed', 
            countResults: countResults ? 'success' : 'failed'
        }, 'SL2 Query Examples Test');
        
        return {
            status: 'completed',
            timestamp: new Date().toISOString()
        };
        
    } catch (error) {
        p.log.error(error, 'Example execution failed');
        return {
            status: 'failed',
            error: error.message
        };
    }
};

// Export for module environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        basicFind,
        comparisonOperators,
        textSearchOperators,
        arrayContainsOperators,
        rtlBasicQueries,
        referenceQueries,
        globalRefQueries,
        basicOrdering,
        basicFieldSelection,
        basicRawQueries,
        basicPagination,
        basicCountOperations,
        basicGrouping,
        bulkUpdateOperations,
        massInsertOperations,
        basicJoins,
        dateTimeBasicQueries,
        iteratorEachExamples,
        complexMultiTableQuery,
        optimizedQueries,
        errorHandlingBasic,
        sqlCasesOperations,
        utilityQueryHelpers,
        integrationWithForms,
        testingQueryPatterns,
        executeExamples
    };
} else {
    // Browser/SL2 environment - run examples
    executeExamples();
}