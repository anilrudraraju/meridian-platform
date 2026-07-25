"""
Synthetic Form 4 XML fixtures mirroring real structures confirmed live
against GEV, INBX, and GRAL during development.
"""

# Real shape: single sale transaction (GEV, Abate Victor)
FORM4_SINGLE_SALE = b"""<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4</documentType>
    <issuer>
        <issuerCik>0001996810</issuerCik>
        <issuerName>GE Vernova Inc.</issuerName>
        <issuerTradingSymbol>GEV</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0002005215</rptOwnerCik>
            <rptOwnerName>Abate Victor</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>1</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
            <officerTitle>Chief Executive Officer, Wind</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common stock</value></securityTitle>
            <transactionDate><value>2026-06-01</value></transactionDate>
            <transactionCoding>
                <transactionCode>S</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>4819</value></transactionShares>
                <transactionPricePerShare><value>948.08</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>1835</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
    <derivativeTable></derivativeTable>
</ownershipDocument>
"""

# Real shape: multiple purchases in one filing, split across custodial
# sub-accounts (INBX, Kayyem Jon Faiz) — confirmed live, 4 transactions.
FORM4_MULTI_PURCHASE = b"""<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4</documentType>
    <issuer>
        <issuerCik>0002007919</issuerCik>
        <issuerName>Inhibrx Biosciences, Inc.</issuerName>
        <issuerTradingSymbol>INBX</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001366268</rptOwnerCik>
            <rptOwnerName>Kayyem Jon Faiz</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>0</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2024-11-19</value></transactionDate>
            <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>34843</value></transactionShares>
                <transactionPricePerShare><value>14.2269</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>51093</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>I</value></directOrIndirectOwnership>
                <natureOfOwnership><value>By Child A</value></natureOfOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2024-11-19</value></transactionDate>
            <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>34843</value></transactionShares>
                <transactionPricePerShare><value>14.2269</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>51093</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>I</value></directOrIndirectOwnership>
                <natureOfOwnership><value>By Child B</value></natureOfOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""

# Real shape: option exercise (code 'M') — increases shares owned but is
# NOT an open-market purchase; must be excluded from insider-buying counts.
FORM4_OPTION_EXERCISE = b"""<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4</documentType>
    <issuer>
        <issuerCik>0001996810</issuerCik>
        <issuerName>GE Vernova Inc.</issuerName>
        <issuerTradingSymbol>GEV</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0009999999</rptOwnerCik>
            <rptOwnerName>Example Person</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>1</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common stock</value></securityTitle>
            <transactionDate><value>2024-05-01</value></transactionDate>
            <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1000</value></transactionShares>
                <transactionPricePerShare><value>0</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>5000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""

# Confirmed live: a filing can have an entirely empty nonDerivativeTable
# (all activity was derivative-only, e.g. an option grant).
FORM4_EMPTY_NONDERIVATIVE = b"""<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4</documentType>
    <issuer>
        <issuerCik>0001996810</issuerCik>
        <issuerName>GE Vernova Inc.</issuerName>
        <issuerTradingSymbol>GEV</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0008888888</rptOwnerCik>
            <rptOwnerName>Another Person</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>0</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable></nonDerivativeTable>
    <derivativeTable>
        <derivativeTransaction>
            <securityTitle><value>Stock Option</value></securityTitle>
        </derivativeTransaction>
    </derivativeTable>
</ownershipDocument>
"""
